
import os
import os.path as osp
from cv2 import split
import numpy as np
import json
import cv2
import base64
import imageio
import torch
from tqdm import tqdm


from metro.modeling._smpl import SMPL
from metro.utils.tsv_file_ops import tsv_writer, generate_linelist_file
import trimesh

from pose_estimation.poses_2d import estimate_2d_joints

#For mesh sampling
from psbody.mesh import Mesh
import mesh_operations


class BCNetDataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.output_dir = 'tsv_datas'
        self.gar_styles=['shirts','short_shirts','pants','short_pants','skirts','short_skirts']
        self.gar_pca_datas={}

        for gar in self.gar_styles:
	        self.gar_pca_datas[gar]=None
        
        self.img_list_file_path = osp.join(data_dir, "motion_datas", "imgfiles.txt")
        assert osp.isfile(self.img_list_file_path)

        with open(self.img_list_file_path, "r") as imgList:
            imgfiles = imgList.read().split("\n")
            self.imgfiles = [osp.join(data_dir, "motion_datas", file) for file in imgfiles]
        
        self.imgs_count = len(imgfiles)

        self.gar_data_by_styles = self.classifyImagesByGarmentStyles()
        
        # change working directory to metro
        this_dir = osp.dirname(__file__)
        metro_dir = osp.join(this_dir, '..', 'libs', 'MeshTransformer')
        cwd = os.getcwd()
        os.chdir(metro_dir)

        self.smpl = SMPL().cuda()
        
        os.chdir(cwd)

        self.test = 0

    def __len__(self):
        return self.imgs_count

    def __getitem__(self, index):

        imgfile = self.imgfiles[index]
        assert osp.isfile(imgfile)
        
        #img_info = imgfile[imgfile.find('SPRING'):imgfile.find('/vmode')]
        #SPRING,gar_up, gar_up_id, gar_bottom, gar_bottom_id, motion = self.decode_info_folder(img_info)
        
        #up_verts = self.pca_verts(gar_up, np.load(osp.join(self.data_dir, gar_up, SPRING, gar_up_id, 'pca_param.npy')))
        #bottom_verts = self.pca_verts(gar_bottom, np.load(osp.join(self.data_dir, gar_bottom, SPRING, gar_bottom_id, 'pca_param.npy')))	
	
        return imgfile

    def decode_info_folder(self, folder):
        
        SPRING=folder.split('_')[0]
        #up
        if 'short_shirts' in folder:
            up='short_shirts'		
        elif 'shirts' in folder:
            up='shirts'
        up_id=folder[folder.find(up)+len(up)+1]
        assert(int(up_id) in [1,2,3])
        #bottom
        if 'short_pants' in folder:
            bottom='short_pants'
        elif 'short_skirts' in folder:
            bottom='short_skirts'
        elif 'pants' in folder:
            bottom='pants'
        elif 'skirts' in folder:
            bottom='skirts'
        
        bottom_id=folder[folder.find(bottom)+len(bottom)+1]
        motion=folder[folder.find(bottom)+len(bottom)+3:]

        info = {}
        info['spring'] = SPRING
        info['gar_up'] = up
        info['gar_up_id'] = up_id
        info['gar_bottom'] = bottom
        info['gar_bottom_id'] = bottom_id
        info['motion'] = motion

        return info

    def pca_verts(self, gartype, pca):
        if self.gar_pca_datas[gartype] is None:
            data=np.load(osp.join(self.data_dir, 'tmps', gartype, 'pca_data.npz'))
            self.gar_pca_datas[gartype]=[data['mean'],data['components']]
        verts=(pca.reshape(1,-1)@self.gar_pca_datas[gartype][1] + self.gar_pca_datas[gartype][0].reshape(1,-1))
        return verts.reshape(-1,3)
    
    #按服裝款式將圖片進行分類
    def classifyImagesByGarmentStyles(self):
        gar_data = {}
        for style in self.gar_styles:
            gar_data[style] = []

        for img_idx in tqdm(range(self.imgs_count)):
            img_file = self.imgfiles[img_idx]
            img_info_str = img_file[img_file.find('SPRING'):img_file.find('/vmode')]
            img_info = self.decode_info_folder(img_info_str)
            gar_data[img_info['gar_up']].append(img_idx)
            gar_data[img_info['gar_bottom']].append(img_idx)

        return gar_data
    
    def generate_garment_template_data(self, gar_style):
        #Generate mesh sampling parameters
        gar_temp_file = osp.join(self.data_dir, 'tmps', gar_style, 'garment_tmp.obj')
        mesh = Mesh(filename=gar_temp_file)
        _, A, D, U = mesh_operations.generate_transform_matrices(mesh=mesh, factors=[4,4,4])
            
        output_dir = osp.join(self.data_dir, self.output_dir, 'garment_templates', gar_style)
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
            
        np.savez(osp.join(output_dir, 'garment_mesh_sampling.npz'), A=A, D=D, U=U)
        mesh.write_obj(filename=osp.join(output_dir, 'garment_mesh.obj'))
        #data = np.load(osp.join(mesh_sampling_dir, 'garment_mesh_sampling.npz'), encoding='latin1', allow_pickle=True)
        #_A = data['A']
        #_D = data['D']
        #_U = data['U']

    # rotateMesh(mesh, [1,0,0], np.pi)
    def rotateMesh(self, mesh, axis, theta):
        #mesh = trimesh.load('')
        newMesh = trimesh.Trimesh(vertices=mesh.v, faces=mesh.f)
        rotMatrix = trimesh.transformations.rotation_matrix(theta, axis)
        newMesh.apply_transform(rotMatrix)

        #scene = trimesh.Scene({'mesh': mesh})
        #print(scene.graph.nodes)
        #scene.show()
        
        return Mesh(v=newMesh.vertices, f=newMesh.faces)


    def pre_process_dataset(self, ds_name, ds_subset, indices):
        
        rows_img, rows_label, rows_hw = [], [], []
        
        tsv_data_dir = osp.join(self.data_dir, "tsv_datas", ds_name, ds_subset) 
        tsv_img_file = osp.join(tsv_data_dir, "imgs.tsv")
        tsv_hw_file = osp.join(tsv_data_dir, "hw.tsv")
        tsv_label_file = osp.join(tsv_data_dir, "labels.tsv")
        tsv_linelist_file = osp.join(tsv_data_dir, "linelist.tsv")

        for img_idx in tqdm(indices):
            img_file = self.imgfiles[img_idx]
            img_path = img_file.split(osp.sep)
            img_name = osp.join(img_path[-3], img_path[-2], img_path[-1])

            #Step 0: encode image
            img = cv2.imread(img_file)
            img_encoded_str = base64.b64encode(cv2.imencode('.jpg', img)[1])
            row_img = [img_name, img_encoded_str]
            rows_img.append(row_img)

            #Step 1: get image size
            rows_hw.append([img_name, 
                            json.dumps([{"height": img.shape[0], "width": img.shape[1]}])
                        ])
            
            #Step 2: get image center and scale


            #Step 3: get 2d joints
            center = [0, 0]
            scale = 0
            scaleFactor = 1.0
            key_joints = np.zeros([17, 2])
            joints_info = estimate_2d_joints(img_file)
            try:
                center = joints_info['center'].tolist()
                scale = scaleFactor * max(joints_info['scale'])
                key_joints = joints_info['joints'][0]
            except KeyError:
                ys, xs = np.where(np.min(img,axis=2)<255)
                bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
                center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200.
                print("no joints detected for image: " + img_name)
                continue
            
            vis = np.ones(17)
            gt_joints_2d = np.zeros([1, 24, 3])
            gt_joints_2d[0, :17, :] = np.hstack([key_joints, np.vstack(vis)])

            #Step 4: get garment and smpl body data
            train_data_path = osp.join(self.data_dir, 'motion_datas/all_train_datas/%d.npz' % img_idx)
            train_data = np.load(train_data_path)
            body_shape = train_data['shape']
            body_shape_t = torch.from_numpy(body_shape).unsqueeze(0).cuda().float()
            body_pose = train_data['pose']
            body_pose_t = torch.from_numpy(body_pose).unsqueeze(0).cuda().float()

            gt_body_vertices = self.smpl(body_pose_t, body_shape_t)
            
            gt_keypoints_3d = self.smpl.get_joints(gt_body_vertices)
            gt_joints_3d = np.asarray(gt_keypoints_3d.cpu())
            gt_joints_3d_tag = np.ones([1, 24, 4])
            gt_joints_3d_tag[0, :, 0:3] = gt_joints_3d


            #for test only
            if self.test == 1:
                output_dir1 = osp.join(self.data_dir, self.output_dir, 'data_tests')
                if not osp.exists(output_dir1):
                    os.makedirs(output_dir1)
                
                human_mesh = Mesh(v=gt_body_vertices[0].cpu().numpy(), f=self.smpl.faces.cpu().numpy())
                human_mesh.write_obj(filename=osp.join(output_dir1, 'human_mesh.obj'))

                human_mesh_r = self.rotateMesh(human_mesh, [1,0,0], np.pi)
                human_mesh_r.write_obj(filename=osp.join(output_dir1, 'human_mesh_r.obj'))
                #end for test only
            
            #Step 5: load garment vertices
            img_info_str = img_file[img_file.find('SPRING'):img_file.find('/vmode')]
            img_info = self.decode_info_folder(img_info_str)

            gar_temp_vertices_up = self.pca_verts(img_info['gar_up'], np.load(osp.join(self.data_dir, 'neutral_datas', img_info['gar_up'], img_info['spring'], img_info['gar_up_id'], 'pca_param.npy')))
            gar_temp_vertices_bottom = self.pca_verts(img_info['gar_bottom'], np.load(osp.join(self.data_dir, 'neutral_datas', img_info['gar_bottom'], img_info['spring'],img_info['gar_bottom_id'], 'pca_param.npy')))	
	
            gar_gt_vertices_up = train_data['up']
            gar_gt_vertices_bottom = train_data['bottom']
            gar_gt_trans = train_data['tran']
            
            #For test only
            gar_temp_file = osp.join(self.data_dir, self.output_dir, 'garment_templates', img_info['gar_up'], 'garment_mesh.obj')
            gar_temp_mesh = Mesh(filename=gar_temp_file)
            gar_up_faces = gar_temp_mesh.f
            gar_mesh_up = Mesh(v=gar_gt_vertices_up, f=gar_up_faces)
            gar_mesh_up_r = self.rotateMesh(gar_mesh_up, [1,0,0], np.pi)

            if self.test == 1:
                gar_temp_mesh.write_obj(filename=osp.join(output_dir1, 'garment_temp_mesh_up.obj'))      
                gar_temp_mesh_r = self.rotateMesh(gar_temp_mesh, [1,0,0], np.pi)
                gar_temp_mesh_r.write_obj(filename=osp.join(output_dir1, 'garment_temp_mesh_up_r.obj'))

                gar_temp_mesh_pca = Mesh(v=gar_temp_vertices_up, f=gar_temp_mesh.f)
                gar_temp_mesh_pca.write_obj(filename=osp.join(output_dir1, 'garment_temp_mesh_up_pca.obj'))
                gar_temp_mesh_pca_r = self.rotateMesh(gar_temp_mesh_pca, [1,0,0], np.pi)
                gar_temp_mesh_pca_r.write_obj(filename=osp.join(output_dir1, 'garment_temp_mesh_up_pca_r.obj'))

                gar_mesh = Mesh(v=gar_gt_vertices_up, f=gar_temp_mesh.f)
                gar_mesh.write_obj(filename=osp.join(output_dir1, 'garment_mesh_up.obj'))
                gar_mesh_r = self.rotateMesh(gar_mesh, [1,0,0], np.pi)
                gar_mesh_r.write_obj(filename=osp.join(output_dir1, 'garment_mesh_up_r.obj'))


            gar_temp_file = osp.join(self.data_dir, self.output_dir, 'garment_templates', img_info['gar_bottom'], 'garment_mesh.obj')
            gar_temp_mesh = Mesh(filename=gar_temp_file)
            gar_bottom_faces = gar_temp_mesh.f
            gar_mesh_bottom = Mesh(v=gar_gt_vertices_bottom, f=gar_bottom_faces)
            gar_mesh_bottom_r = self.rotateMesh(gar_mesh_bottom, [1,0,0], np.pi)

            if self.test == 1:
                gar_temp_mesh.write_obj(filename=osp.join(output_dir1, 'garment_temp_mesh_bottom.obj'))
                gar_temp_mesh_r = self.rotateMesh(gar_temp_mesh, [1,0,0], np.pi)
                gar_temp_mesh_r.write_obj(filename=osp.join(output_dir1, 'garment_temp_mesh_bottom_r.obj'))

                gar_temp_mesh_pca = Mesh(v=gar_temp_vertices_bottom, f=gar_temp_mesh.f)
                gar_temp_mesh_pca.write_obj(filename=osp.join(output_dir1, 'garment_temp_mesh_bottom_pca.obj'))
                gar_temp_mesh_pca_r = self.rotateMesh(gar_temp_mesh_pca, [1,0,0], np.pi)
                gar_temp_mesh_pca_r.write_obj(filename=osp.join(output_dir1, 'garment_temp_mesh_bottom_pca_r.obj'))

                gar_mesh = Mesh(v=gar_gt_vertices_bottom, f=gar_temp_mesh.f)
                gar_mesh.write_obj(filename=osp.join(output_dir1, 'garment_mesh_bottom.obj'))
                gar_mesh_r = self.rotateMesh(gar_mesh, [1,0,0], np.pi)
                gar_mesh_r.write_obj(filename=osp.join(output_dir1, 'garment_mesh_bottom_r.obj'))

            #End for test only

            #Step 6: save to labels
            labels = [{
                "center": center,
                "scale": scale,
                "2d_joints": gt_joints_2d.tolist(),
                "has_2d_joints": 1,
                "3d_joints": gt_joints_3d_tag.tolist(),
                "has_3d_joints": 1,
                "betas": body_shape.tolist(),
                "pose": body_pose.tolist(),
                "has_smpl": 1,
                "garment": {
                    "style_up": img_info['gar_up'],
                    "style_bottom": img_info['gar_bottom'],
                    "temp_verts_up": gar_temp_vertices_up.tolist(),
                    "temp_verts_bottom": gar_temp_vertices_bottom.tolist(),
                    "faces_up": gar_up_faces.tolist(),
                    "verts_up": gar_gt_vertices_up.tolist(),
                    "verts_up_r": gar_mesh_up_r.v.tolist(),
                    "faces_bottom": gar_bottom_faces.tolist(),
                    "verts_bottom": gar_gt_vertices_bottom.tolist(),
                    "verts_bottom_r": gar_mesh_bottom_r.v.tolist(),
                    "verts_trans": gar_gt_trans.tolist()
                }
            }]

            row_label = [img_name, json.dumps(labels)]
            rows_label.append(row_label)

        #Save to files
        if not osp.exists(tsv_data_dir):
            os.makedirs(tsv_data_dir)

        tsv_writer(rows_img, tsv_img_file)
        tsv_writer(rows_label, tsv_label_file)
        tsv_writer(rows_hw, tsv_hw_file)

        # generate linelist file
        generate_linelist_file(tsv_label_file, save_file=tsv_linelist_file)

    def pre_process(self, catalog, dataset_size=None):

        #for gar in self.gar_styles:
        #    self.generate_garment_template_data(gar)
        
        train_set = []
        test_set = []

        if catalog == 'all':
            all_size = 2000 #self.imgs_count
            train_size = (int)(0.8 * all_size)
            test_size = all_size - train_size
            train_split, test_split = torch.utils.data.random_split(range(all_size), [train_size, test_size])
            train_set = train_split.indices
            test_set = test_split.indices
        else:
            #Split dataset
            dataset = np.array(self.gar_data_by_styles[catalog])
            all_size = None
            if dataset_size != None:
                dataset = dataset[:dataset_size]
                all_size = dataset_size
            else:
                all_size = len(dataset)

            train_size = (int)(0.8 * all_size)
            test_size = all_size - train_size
            train_split, test_split = torch.utils.data.random_split(dataset, [train_size, test_size])
            train_set = dataset[train_split.indices]
            test_set = dataset[test_split.indices]

        self.pre_process_dataset(catalog, 'train', train_set)
        self.pre_process_dataset(catalog, 'test', test_set)

def main_test():
    #dataset = BCNetDataset("/home/guomin/datadisk/Projects/Research/BCNet/body_garment_dataset")
    dataset = BCNetDataset("/media/guomin/Works/Projects/Research/1-BCNet/body_garment_dataset")
    dataset.pre_process('shirts', 2000)

if __name__ == "__main__":
    main_test()
