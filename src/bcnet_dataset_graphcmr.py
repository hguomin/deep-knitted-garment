#Guomin Huang @2022.09.12

import os
import os.path as osp
import cv2
from cv2 import split
import numpy as np
import json

from tqdm import tqdm

# for mesh sampling and rotation
import trimesh
from psbody.mesh import Mesh
import mesh_operations

# for 2d poses detection
import pose_estimation.openpose as pose

class BCNetDataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.gar_styles=['shirts','short_shirts','pants','short_pants','skirts','short_skirts']
        self.gar_pca_datas={}

        for gar in self.gar_styles:
	        self.gar_pca_datas[gar]=None
        
        self.img_list_file_path = osp.join(data_dir, "motion_datas", "imgfiles.txt")
        assert osp.isfile(self.img_list_file_path)

        with open(self.img_list_file_path, "r") as imgList:
            imgfiles = imgList.read().split("\n")
            self.imgfiles = np.array([osp.join("motion_datas", file) for file in imgfiles])
        
        self.imgs_count = len(self.imgfiles)
        self.img_info_list = self.generate_images_info_list()
        self.img_info_by_styles = self.classify_images_by_garment_styles()

    def count(self):
        return self.imgs_count

    def image_absolute_path(self, img_idx):
        return osp.join(self.data_dir, self.imgfiles[img_idx])
        
    def image_list_by_garment_style(self, gar_style):
        return self.img_info_by_styles[gar_style]

    def image_names(self, img_idx):
        return self.imgfiles[img_idx]

    # 衣服的原始三維網格
    def garment_3d_template_data(self, gar_style):
        return osp.join(self.data_dir, 'tmps', gar_style, 'garment_tmp.obj')

    # 衣服的PCA變形的參數，用於原始三維網格的變形
    def garment_pca_data(self, gar_style, spring_id, gar_style_id):
        return np.load(osp.join(self.data_dir, 'neutral_datas', gar_style, spring_id, gar_style_id, 'pca_param.npy'))

    def garment_pca_data_by_img_id(self, img_idx):
        img_info = self.img_info_list[img_idx]

        #np.load(osp.join(self.data_dir, 'neutral_datas', img_info['gar_up'], img_info['spring'], img_info['gar_up_id'], 'pca_param.npy'))
        gar_pca_up =  self.garment_pca_data(img_info['gar_up'], img_info['spring'], img_info['gar_up_id'])
        gar_pca_bottom =  self.garment_pca_data(img_info['gar_bottom'], img_info['spring'], img_info['gar_bottom_id'])

        return gar_pca_up, gar_pca_bottom

    # 衣服模板經PCA變形後的頂點數據
    def garment_pca_verts(self, gar_style, pca):
        if self.gar_pca_datas[gar_style] is None:
            data=np.load(osp.join(self.data_dir, 'tmps', gar_style, 'pca_data.npz'))
            self.gar_pca_datas[gar_style]=[data['mean'],data['components']]
        verts=(pca.reshape(1,-1)@self.gar_pca_datas[gar_style][1] + self.gar_pca_datas[gar_style][0].reshape(1,-1))
        return verts.reshape(-1,3)
    
    # 根据图片id获取图片中上下衣服的PCA模板数据
    def garment_pca_template_verts(self, img_idx):
        pca_up, pca_bottom = self.garment_pca_data_by_img_id(img_idx)
        pca_verts_up = self.garment_pca_verts(self.img_info_list[img_idx]['gar_up'], pca_up)
        pca_verts_bottom = self.garment_pca_verts(self.img_info_list[img_idx]['gar_bottom'], pca_bottom)

        return pca_verts_up, pca_verts_bottom

    # 圖片對應的標籤數據:
    # up: 上衣服裝網格頂點
    # bottom: 下衣服裝網格頂點
    # tran: 服裝網格的偏移，服裝網格頂點坐標需要加上這個偏移才能穿到人體上。
    # pose: 人體姿態參數
    # shape: 人體形態參數
    def train_annotation_data(self, img_idx): # img_idx: the index of imgage in the images list file(motion_datas/imgfiles.txt)
        train_data_path = osp.join(self.data_dir, 'motion_datas/all_train_datas/%d.npz' % img_idx)
        train_data = np.load(train_data_path)
        return train_data

    def decode_garment_info(self, img_file):
        folder = img_file[img_file.find('SPRING'):img_file.find('/vmode')]

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

    # 建立衣服款式信息的列表，通過id可以快速獲取款式信息，此id與圖片列表文件(motion_datas/imgfiles.txt)中的圖片順序對應
    def generate_images_info_list(self):
        gar_infos = []
        for img_idx in tqdm(range(self.imgs_count)):
            img_file = self.imgfiles[img_idx]
            img_info = self.decode_garment_info(img_file)
            gar_infos.append(img_info)

        return np.array(gar_infos)

    #按服裝款式將圖片進行分類
    def classify_images_by_garment_styles(self):
        gar_data = {}
        for style in self.gar_styles:
            gar_data[style] = []

        for img_idx in tqdm(range(self.imgs_count)):
            img_info = self.img_info_list[img_idx] #self.decode_garment_info(img_file)
            gar_data[img_info['gar_up']].append(img_idx)
            gar_data[img_info['gar_bottom']].append(img_idx)

        return gar_data
    
    #獲取所有圖片中人體的二位姿態數據
    def generate_body_2d_joints(self):
        all_joints = list()
        all_imgs = list()
        for img_idx in tqdm(range(self.imgs_count)):
            body_2d_joints, img = self.generate_single_body_2d_joints(img_idx)
            all_joints.append(body_2d_joints)
            all_imgs.append(img)
        return all_joints, all_imgs

    def generate_single_body_2d_joints(self, img_idx):
        img_file = self.image_absolute_path(img_idx)
        return pose.estimate_2d_joints(img_file)

    def save(self):
        # Garment template mesh sampling datas
        for style in self.gar_styles:
            gar_temp_file = osp.join(self.data_dir, 'tmps', style, 'garment_tmp.obj')
            mesh = self.rotateMesh(trimesh.load(gar_temp_file), [1,0,0], np.pi) # Rotate the mesh pi around x axis
            _, A, D, U = mesh_operations.generate_transform_matrices(mesh=Mesh(mesh.vertices,mesh.faces) , factors=[4,4,4])
            mesh.export(osp.join(self.data_dir, 'tmps', style, 'garment_tmp_rotated.obj'))
            np.savez(osp.join(self.data_dir, 'tmps', style, 'garment_mesh_sampling.npz'), A=A, D=D, U=U)

        np.save(osp.join(self.data_dir, 'motion_datas', 'img_garment_info_list.npy'), self.img_info_list, allow_pickle=True)
        np.save(osp.join(self.data_dir, 'motion_datas', 'img_garment_catelog.npy'), [self.img_info_by_styles], allow_pickle=True)

        # 2d joints and garment template vertices
        joints_dir = osp.join(self.data_dir, 'motion_datas', 'all_train_datas_joints')
        temp_verts_dir = osp.join(self.data_dir, 'motion_datas', 'garment_template_pca_vertices')
        for dir in [joints_dir,temp_verts_dir]:
            if not osp.exists(dir):
                os.makedirs(dir)
        for img_id in tqdm(range(self.count())):
            # Get and save the 2d joints in the image
            joints, cv_img = self.generate_single_body_2d_joints(img_id)
            np.save(osp.join(joints_dir, f'{img_id}.npy'), joints)
            # cv2.imwrite(osp.join(joints_dir, f'{img_id}.png'), cv_img)

            # Get and save the pca template vertices of the garment mesh for the image
            pca_temp_verts_up, pca_temp_verts_bottom = self.garment_pca_template_verts(img_id)
            np.savez(osp.join(temp_verts_dir, f'{img_id}.npz'), gar_up=pca_temp_verts_up, gar_bottom=pca_temp_verts_bottom)
            

    def generate_garment_template_data(self, gar_style):
        #Generate mesh sampling parameters
        gar_temp_file = osp.join(self.data_dir, 'tmps', gar_style, 'garment_tmp.obj')
        mesh = Mesh(filename=gar_temp_file)
        _, A, D, U = mesh_operations.generate_transform_matrices(mesh=mesh, factors=[4,4,4])
            
        #output_dir = osp.join(self.data_dir, self.output_dir, 'garment_templates', gar_style)
        #if not osp.exists(output_dir):
        #    os.makedirs(output_dir)
            
        #np.savez(osp.join(output_dir, 'garment_mesh_sampling.npz'), A=A, D=D, U=U)
        #mesh.write_obj(filename=osp.join(output_dir, 'garment_mesh.obj'))
        
        #data = np.load(osp.join(mesh_sampling_dir, 'garment_mesh_sampling.npz'), encoding='latin1', allow_pickle=True)
        #_A = data['A']
        #_D = data['D']
        #_U = data['U']

    # rotateMesh(mesh, [1,0,0], np.pi)
    def rotateMesh2(self, mesh, axis, theta):
        #mesh = trimesh.load('')
        newMesh = trimesh.Trimesh(vertices=mesh.v, faces=mesh.f)
        rotMatrix = trimesh.transformations.rotation_matrix(theta, axis)
        newMesh.apply_transform(rotMatrix)

        #scene = trimesh.Scene({'mesh': mesh})
        #print(scene.graph.nodes)
        #scene.show()
        
        return Mesh(v=newMesh.vertices, f=newMesh.faces)

    def rotateMesh(self, mesh, axis, theta):
        #mesh = trimesh.load('')
        rotMatrix = trimesh.transformations.rotation_matrix(theta, axis)
        mesh.apply_transform(rotMatrix)
        
        return mesh

def ensure_dir_exists(path):
    if not osp.exists(path):
        os.makedirs(path)

def main_test():

    #dataset = BCNetDataset("/home/guomin/datadisk/Projects/Research/BCNet/body_garment_dataset")
    dataset = BCNetDataset("/media/guomin/Works/Projects/Research/1-BCNet/body_garment_dataset")
    dataset.save()

    np.save(osp.join(dataset.data_dir, 'motion_datas', 'img_garment_info_list.npy'), dataset.img_info_list, allow_pickle=True)
    np.save(osp.join(dataset.data_dir, 'motion_datas', 'img_garment_catelog.npy'), [dataset.img_info_by_styles], allow_pickle=True)
    img_info_list = np.load(osp.join(dataset.data_dir, 'motion_datas', 'img_garment_info_list.npy'), allow_pickle=True)
    img_info = img_info_list[100]
    img_info_by_styles = np.load(osp.join(dataset.data_dir, 'motion_datas', 'img_garment_catelog.npy'), allow_pickle=True)
    shirts = img_info_by_styles[0]['shirts']
    # Generate 2d joints for each image and save it
    joints_dir = osp.join(dataset.data_dir, 'motion_datas', 'all_train_datas_joints')
    ensure_dir_exists(joints_dir)
    for img_id in tqdm(range(dataset.count())):
        joints, cv_img = dataset.generate_single_body_2d_joints(img_id)
        np.save(osp.join(joints_dir, f'{img_id}.npy'), joints)
        #cv2.imwrite(osp.join(joints_dir, f'{img_id}.png'), cv_img)



    gar_imgs_all = dataset.image_list_by_garment_style('shirts')
    gar_imgs_ids = np.random.choice(a=gar_imgs_all, size=10000, replace=False)
    img_names_ = dataset.image_names(gar_imgs_ids)

    body_joints_2d = []
    for img_id in tqdm(gar_imgs_ids):
        joints_, cv2_img = dataset.generate_single_body_2d_joints(img_id)
        body_joints_2d.append(joints_)
  
    np.savez(osp.join(dataset.data_dir, 'bcnet_train.npz'), img_ids=gar_imgs_ids, joints_2d=body_joints_2d)
    
    print('ok')

if __name__ == "__main__":
    main_test()
