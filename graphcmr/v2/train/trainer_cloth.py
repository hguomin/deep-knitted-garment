# Guomin @2022/10/01

"""
This file includes the full training procedure.
"""
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torchvision.utils import make_grid
from models.graph_cnn_cloth import ClothGraphCNN3

from utils import BaseTrainer, Mesh, MeshSampler
from datasets import create_dataset
from models import ClothGraphCNN, ClothGraphCNN2, ClothGraphConvNetwork, ClothGraphConvNetwork_MLPDecoder, ClothGraphConvNetwork_MLPDecoder_Fusion, SMPLParamRegressor, SMPL, resnet50
from models.geometric_layers import orthographic_projection, rodrigues
from utils.renderer import Renderer, visualize_reconstruction, visualize_reconstructed_garment


#Guomin
import trimesh
import os.path as osp

class ClothTrainer(BaseTrainer):
    """Trainer object.
    Inherits from BaseTrainer that sets up logging, saving/restoring checkpoints etc.
    """
    def init_fn(self):
        # create training dataset
        self.train_ds = create_dataset(self.options.dataset, self.options)

        # create Mesh object
        self.mesh = Mesh()
        self.faces = self.mesh.faces.to(self.device)

        data_dir = self.train_ds.ds.img_dir
        if self.options.dataset == 'bcnet':
            gar_temp_file = osp.join(data_dir, 'tmps', self.options.garment, 'garment_tmp.obj')
            gar_sampling_params_file = osp.join(data_dir, 'tmps', self.options.garment, 'garment_mesh_sampling.npz')
            gar_mesh = trimesh.load(gar_temp_file)
            self.gar_mesh_sampler = MeshSampler(gar_mesh.vertices, gar_mesh.faces, gar_sampling_params_file)
        
        self.resnet = resnet50(pretrained=True).to(self.device)

        # create GraphCNN
        self.body_gcn = ClothGraphCNN3(self.mesh.adjmat,
                           self.mesh.ref_vertices.t(),
                           num_channels=self.options.num_channels,
                           num_layers=self.options.num_layers
                        ).to(self.device)
        if self.options.cloth_decoder == 'mlp':
            self.cloth_gcn = ClothGraphConvNetwork_MLPDecoder(self.gar_mesh_sampler, num_layers=self.options.num_layers, num_channels=self.options.num_channels).to(self.device)
        elif self.options.cloth_decoder == 'gcn':
            self.cloth_gcn = ClothGraphConvNetwork(self.gar_mesh_sampler, num_layers=self.options.num_layers, num_channels=self.options.num_channels).to(self.device)
        else:
            self.cloth_gcn = ClothGraphConvNetwork_MLPDecoder_Fusion(self.mesh.adjmat, self.gar_mesh_sampler,
                            num_channels=self.options.num_channels,
                            num_layers=self.options.num_layers
                            ).to(self.device)

        # SMPL Parameter regressor
        self.smpl_param_regressor = SMPLParamRegressor().to(self.device)

        # Setup a joint optimizer for the models
        params=list(self.resnet.parameters()) + list(self.body_gcn.parameters()) + list(self.cloth_gcn.parameters()) + list(self.smpl_param_regressor.parameters())
        self.optimizer = torch.optim.Adam( params=params,
                                           lr=self.options.lr,
                                           betas=(self.options.adam_beta1, 0.999),
                                           weight_decay=self.options.wd)

        # SMPL model
        self.smpl = SMPL().to(self.device)

        # Create loss functions
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)

        # Pack models and optimizers in a dict - necessary for checkpointing
        self.models_dict = {'resnet': self.resnet, 'body_gcn': self.body_gcn, 'cloth_gcn': self.cloth_gcn, 'smpl_param_regressor': self.smpl_param_regressor}
        self.optimizers_dict = {'optimizer': self.optimizer}
        
        # Renderer for visualization
        self.renderer = Renderer(faces=self.smpl.faces.cpu().numpy())

        # LSP indices from full list of keypoints
        self.to_lsp = list(range(14))

        # Optionally start training from a pretrained checkpoint
        # Note that this is different from resuming training
        # For the latter use --resume
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d):
        """Compute 2D reprojection loss on the keypoints.
        The confidence is binary and indicates whether the keypoints exist or not.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence
        """
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        """Compute SMPL parameter loss for the examples that SMPL annotations are available."""
        pred_rotmat_valid = pred_rotmat[has_smpl == 1].view(-1, 3, 3)
        gt_rotmat_valid = rodrigues(gt_pose[has_smpl == 1].view(-1,3))
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def train_step(self, input_batch):
        """Training step."""
        # do we need resnet.train()? yes
        self.resnet.train()
        self.body_gcn.train()
        self.cloth_gcn.train()
        self.smpl_param_regressor.train()

        # Grab data from the batch
        gt_keypoints_2d = input_batch['keypoints']
        gt_keypoints_3d = input_batch['pose_3d']
        gt_pose = input_batch['pose']
        gt_betas = input_batch['betas']
        has_smpl = input_batch['has_smpl']
        has_pose_3d = input_batch['has_pose_3d']
        images = input_batch['img']

        gar_gt_vertices = input_batch['gar_vs']
        gar_gt_tran = input_batch['gar_tran']

        # Render vertices using SMPL parameters
        gt_vertices = self.smpl(gt_pose, gt_betas)
        batch_size = gt_vertices.shape[0]

        # Guomin
        # human_mesh = trimesh.Trimesh(vertices=gt_vertices[0].cpu().numpy(), faces=self.smpl.faces.cpu().numpy())
        # human_mesh.export('/media/guomin/Works/Projects/Research/GraphCMR/GraphCMR/logs/human_mesh.obj')

        # Feed image in the GraphCNN
        # Returns subsampled body mesh and camera parameters
        images_resnet = self.resnet(images)
        pred_vertices_sub, pred_camera = self.body_gcn(images_resnet)

        # Upsample mesh in the original size
        pred_vertices = self.mesh.upsample(pred_vertices_sub.transpose(1,2))
        
        # Guomin: Returns subsampled cloth mesh
        #pred_gar_vertices_sub = self.cloth_gcn(images_resnet)
        #pred_gar_vertices = self.gar_mesh_sampler.upsample(pred_gar_vertices_sub.transpose(1,2))
        ##self.gar_mesh_sampler.save_mesh('/media/guomin/Works/Projects/Research/GraphCMR/GraphCMR/logs/gar_mesh.obj', pred_gar_vertices[0].detach())
        
        # Prepare input for SMPL Parameter regressor
        # The input is the predicted and template vertices subsampled by a factor of 4
        # Notice that we detach the GraphCNN
        x = pred_vertices_sub.transpose(1,2).detach()
        x = torch.cat([x, self.mesh.ref_vertices[None, :, :].expand(batch_size, -1, -1)], dim=-1)

        # Guomin
        # pytorch多分支网络：https://blog.csdn.net/Taylent/article/details/107339505
        # 多任务学习：https://blog.csdn.net/tMb8Z9Vdm66wH68VX1/article/details/127130592
        # 是不是也可以参考smpl_param_regressor的方式，detach pred_vertices_sub然后基于预测的SMPL人体模型网格顶点应用n-cloth的方法再去预测衣服模型？

        # Estimate SMPL parameters and render vertices
        pred_rotmat, pred_shape = self.smpl_param_regressor(x)
        pred_vertices_smpl = self.smpl(pred_rotmat, pred_shape)

        # Guomin: use fusion networks to predict cloth mesh
        body_vertices_smpl = pred_vertices_smpl.detach()
        body_vertices_smpl_sub = self.mesh.downsample(body_vertices_smpl).transpose(1,2)

        if self.options.cloth_decoder == 'mlp' or self.options.cloth_decoder == 'gcn':
            pred_gar_vertices_sub = self.cloth_gcn(images_resnet)
        else:
            pred_gar_vertices_sub = self.cloth_gcn(images_resnet, body_vertices_smpl_sub, self.mesh.adjmat)


        pred_gar_vertices = self.gar_mesh_sampler.upsample(pred_gar_vertices_sub.transpose(1,2))

        # Get 3D and projected 2D keypoints from the regressed shape
        pred_keypoints_3d = self.smpl.get_joints(pred_vertices)
        pred_keypoints_2d = orthographic_projection(pred_keypoints_3d, pred_camera)[:, :, :2]
        pred_keypoints_3d_smpl = self.smpl.get_joints(pred_vertices_smpl)
        pred_keypoints_2d_smpl = orthographic_projection(pred_keypoints_3d_smpl, pred_camera.detach())[:, :, :2]

        # Compute losses

        # GraphCNN losses 
        loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, has_pose_3d)
        loss_shape = self.shape_loss(pred_vertices, gt_vertices, has_smpl)

        # SMPL regressor losses
        loss_keypoints_smpl = self.keypoint_loss(pred_keypoints_2d_smpl, gt_keypoints_2d)
        loss_keypoints_3d_smpl = self.keypoint_3d_loss(pred_keypoints_3d_smpl, gt_keypoints_3d, has_pose_3d)
        loss_shape_smpl = self.shape_loss(pred_vertices_smpl, gt_vertices, has_smpl)
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_shape, gt_pose, gt_betas, has_smpl)                                                

        # Guomin: garment shape loss
        gar_gt_vertices = gar_gt_vertices - gar_gt_tran
        loss_shape_garment = self.shape_loss(pred_gar_vertices, gar_gt_vertices, has_smpl)

        # Add losses to compute the total loss
        loss = loss_shape_smpl + loss_keypoints_smpl + loss_keypoints_3d_smpl +\
               loss_regr_pose + 0.1 * loss_regr_betas + loss_shape + loss_keypoints + loss_keypoints_3d +\
               loss_shape_garment

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pack output arguments to be used for visualization in a list
        out_args = [pred_vertices, pred_vertices_smpl, pred_gar_vertices, pred_camera, 
                    pred_keypoints_2d, pred_keypoints_2d_smpl,
                    loss_shape, loss_shape_smpl ,loss_shape_garment, loss_keypoints, loss_keypoints_smpl,
                    loss_keypoints_3d, loss_keypoints_3d_smpl,
                    loss_regr_pose, loss_regr_betas, loss]
        out_args = [arg.detach() for arg in out_args]
        return out_args

    def train_summaries(self, input_batch,
                        pred_vertices, pred_vertices_smpl, pred_vertices_garment, pred_camera,
                        pred_keypoints_2d, pred_keypoints_2d_smpl,
                        loss_shape, loss_shape_smpl, loss_shape_garment, loss_keypoints, loss_keypoints_smpl,
                        loss_keypoints_3d, loss_keypoints_3d_smpl,
                        loss_regr_pose, loss_regr_betas, loss):
        """Tensorboard logging."""
        gt_keypoints_2d = input_batch['keypoints'].cpu().numpy()
         
        rend_imgs = []
        rend_imgs_smpl = []
        # Guomin: draw garment
        rend_imgs_garment = []

        batch_size = pred_vertices.shape[0]
        # Do visualization for the first 4 images of the batch
        for i in range(min(batch_size, 4)):
            img = input_batch['img_orig'][i].cpu().numpy().transpose(1,2,0)
            # Get LSP keypoints from the full list of keypoints
            gt_keypoints_2d_ = gt_keypoints_2d[i, self.to_lsp]
            pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, self.to_lsp]
            pred_keypoints_2d_smpl_ = pred_keypoints_2d_smpl.cpu().numpy()[i, self.to_lsp]
            # Get GraphCNN and SMPL vertices for the particular example
            vertices = pred_vertices[i].cpu().numpy()
            vertices_smpl = pred_vertices_smpl[i].cpu().numpy()
            cam = pred_camera[i].cpu().numpy()
            cam = pred_camera[i].cpu().numpy()
            # Visualize reconstruction and detected pose
            rend_img = visualize_reconstruction(img, self.options.img_res, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, self.renderer)
            rend_img_smpl = visualize_reconstruction(img, self.options.img_res, gt_keypoints_2d_, vertices_smpl, pred_keypoints_2d_smpl_, cam, self.renderer)
            
            # Guomin: draw garment
            vertices_garment = pred_vertices_garment[i].cpu().numpy()
            #rend_img_garment = visualize_reconstructed_garment(rend_img, self.options.img_res, gt_keypoints_2d_, vertices_garment, self.gar_mesh_sampler.faces.cpu().numpy(), pred_keypoints_2d_smpl_, cam, self.renderer)
            rend_img_garment = visualize_reconstructed_garment(img, self.options.img_res, gt_keypoints_2d_, vertices_garment, self.gar_mesh_sampler.faces.cpu().numpy(), pred_keypoints_2d_smpl_, cam, self.renderer)
            rend_img_garment = rend_img_garment.transpose(2,0,1)

            rend_img = rend_img.transpose(2,0,1)
            rend_img_smpl = rend_img_smpl.transpose(2,0,1)
            rend_imgs.append(torch.from_numpy(rend_img))
            rend_imgs_smpl.append(torch.from_numpy(rend_img_smpl))

            # Guomin: draw garment
            rend_imgs_garment.append(torch.from_numpy(rend_img_garment))

        rend_imgs = make_grid(rend_imgs, nrow=1)
        rend_imgs_smpl = make_grid(rend_imgs_smpl, nrow=1)

        # Guomin: draw garment
        rend_imgs_garment = make_grid(rend_imgs_garment, nrow=1)

        # Save results in Tensorboard
        self.summary_writer.add_image('imgs', rend_imgs, self.step_count)
        self.summary_writer.add_image('imgs_smpl', rend_imgs_smpl, self.step_count)
        
        # Guomin: draw garment
        self.summary_writer.add_image('imgs_garment', rend_imgs_garment, self.step_count)

        self.summary_writer.add_scalar('loss_shape', loss_shape, self.step_count)
        self.summary_writer.add_scalar('loss_shape_smpl', loss_shape_smpl, self.step_count)
        self.summary_writer.add_scalar('loss_shape_garment', loss_shape_garment, self.step_count)
        self.summary_writer.add_scalar('loss_regr_pose', loss_regr_pose, self.step_count)
        self.summary_writer.add_scalar('loss_regr_betas', loss_regr_betas, self.step_count)
        self.summary_writer.add_scalar('loss_keypoints', loss_keypoints, self.step_count)
        self.summary_writer.add_scalar('loss_keypoints_smpl', loss_keypoints_smpl, self.step_count)
        self.summary_writer.add_scalar('loss_keypoints_3d', loss_keypoints_3d, self.step_count)
        self.summary_writer.add_scalar('loss_keypoints_3d_smpl', loss_keypoints_3d_smpl, self.step_count)
        self.summary_writer.add_scalar('loss', loss, self.step_count)
