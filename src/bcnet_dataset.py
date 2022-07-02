
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

class BCNetDataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.gartypes=['shirts','short_shirts','pants','short_pants','skirts','short_skirts']
        self.gar_pca_datas={}
        for gar in self.gartypes:
	        self.gar_pca_datas[gar]=None
        
        self.img_list_file_path = osp.join(data_dir, "motion_datas", "imgfiles.txt")
        assert osp.isfile(self.img_list_file_path)

        with open(self.img_list_file_path, "r") as imgList:
            imgfiles = imgList.read().split("\n")
            self.imgfiles = [osp.join(data_dir, "motion_datas", file) for file in imgfiles]
        
        self.imgs_count = len(imgfiles)

        self.smpl = SMPL().cuda()

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
            img = imageio.imread(img_file)
            ys, xs = np.where(np.min(img,axis=2)<255)
            bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
            center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
            scaleFactor = 1.2
            scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200.

            #Step 3: get 2d joints, to do...
            gt_joints_2d = np.zeros([1, 24, 3])

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

            #Step 5: save to labels
            labels = [{
                "center": center,
                "scale": scale,
                "2d_joints": gt_joints_2d.tolist(),
                "has_2d_joints": 0,
                "3d_joints": gt_joints_3d_tag.tolist(),
                "has_3d_joints": 1,
                "betas": body_shape.tolist(),
                "pose": body_pose.tolist(),
                "has_smpl": 1
            }]

            row_label = [img_name, json.dumps(labels)]
            rows_label.append(row_label)

            #gar_vertices_up = train_data['up']
            #gar_vertices_bottom = train_data['bottom']
            #trans = train_data['tran']
            
            #img_info_str = img_file[img_file.find('SPRING'):img_file.find('/vmode')]
            #img_info = self.decode_info_folder(img_info_str)

        #Save to files
        if not osp.exists(tsv_data_dir):
            os.makedirs(tsv_data_dir)

        tsv_writer(rows_img, tsv_img_file)
        tsv_writer(rows_label, tsv_label_file)
        tsv_writer(rows_hw, tsv_hw_file)

        # generate linelist file
        generate_linelist_file(tsv_label_file, save_file=tsv_linelist_file)


    def pre_process(self):
        all_size = self.imgs_count
        train_size = (int)(0.8 * all_size)
        test_size = all_size - train_size
        train_split, test_split = torch.utils.data.random_split(range(all_size), [train_size, test_size])

        self.pre_process_dataset('all', 'train', train_split.indices)
        self.pre_process_dataset('all', 'test', test_split.indices)

def main_test():
    dataset = BCNetDataset("/home/guomin/datadisk/Projects/Research/BCNet/body_garment_dataset")
    dataset.pre_process()

if __name__ == "__main__":
    main_test()
