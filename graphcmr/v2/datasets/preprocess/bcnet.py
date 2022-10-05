# Guomin @2022.09.17
import os
import os.path as osp
import numpy as np
from tqdm import tqdm

def bbox_from_openpose(keypoints, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale

def bcnet_extract(dataset_path, out_path, gar_style):

    # The joints is generated with openpose COCO joints model, need to map it to GraphCMR's definition
    # openpose COCO: 
    #    {0,  "Nose"},
    #    {1,  "Neck"},
    #    {2,  "RShoulder"},
    #    {3,  "RElbow"},
    #    {4,  "RWrist"},
    #    {5,  "LShoulder"},
    #    {6,  "LElbow"},
    #    {7,  "LWrist"},
    #    {8,  "RHip"},
    #    {9,  "RKnee"},
    #    {10, "RAnkle"},
    #    {11, "LHip"},
    #    {12, "LKnee"},
    #    {13, "LAnkle"},
    #    {14, "REye"},
    #    {15, "LEye"},
    #    {16, "REar"},
    #    {17, "LEar"},
    #    {18, "Background"}
    joints_map = [19, 12, 8, 7, 6, 9, 10, 11, 2, 1, 0, 3, 4, 5, 21, 20, 23, 22]

    img_list_file_path = osp.join(dataset_path, "motion_datas", "imgfiles.txt")
    assert osp.isfile(img_list_file_path)

    with open(img_list_file_path, "r") as imgList:
        all_imgs_list = imgList.read().split("\n")
        all_imgs_list = [osp.join("motion_datas", file) for file in all_imgs_list]

    # garment images in a same styles    
    img_info_by_styles = np.load(osp.join(dataset_path, 'motion_datas', 'img_garment_catelog.npy'), allow_pickle=True)
    gar_img_ids = img_info_by_styles[0][gar_style]

    imgs, imgs_name_list, centers_list, scales_list, parts_list, poses_list, shapes_list, gar_up_list, gar_bottom_list, gar_tran_list = [], [], [], [], [], [], [], [], [], []
    ss = 0
    for img_id in tqdm(gar_img_ids):
        # key joints 
        key_joints = np.load(osp.join(dataset_path, 'motion_datas', 'all_train_datas_joints', f'{img_id}.npy'))
        # ignore the images that contains multiple persons
        if (key_joints.shape[0] > 1):
            ss = ss + 1
            continue
        key_joints = key_joints[0]

        # center and scale
        center, scale = bbox_from_openpose(key_joints)
        centers_list.append(center)
        scales_list.append(scale)

        # key joints 
        key_joints[key_joints[:,2]>0,2] = 1
        parts = np.zeros([24, 3])
        parts[joints_map] = key_joints
        parts_list.append(parts)

        # image ids
        imgs.append(img_id)

        # image name
        imgs_name_list.append(all_imgs_list[img_id])

        # human body label data
        labels_data = np.load(osp.join(dataset_path, 'motion_datas', 'all_train_datas', f'{img_id}.npz'))
        poses_list.append(labels_data['pose'])
        shapes_list.append(labels_data['shape'])
        gar_up_list.append(labels_data['up'])
        gar_bottom_list.append(labels_data['bottom'])
        gar_tran_list.append(labels_data['tran'])
        
    # save to output folder
    extra_path = osp.join(out_path, 'extras')
    if not osp.isdir(extra_path):
        os.makedirs(extra_path)

    np.savez(osp.join(extra_path, 'bcnet_shirts.npz'),
            img_ids=imgs,
            imgname=imgs_name_list,
            center=centers_list,
            scale=scales_list,
            part=parts_list,
            pose=poses_list,
            shape=shapes_list)
            # the file is too large if we add below datas:
            #gar_up=gar_up_list,
            #gar_bottom=gar_bottom_list,
            #gar_tran=gar_tran_list)
                
    print(ss)

def main_test():
    bcnet_extract('/media/guomin/Works/Projects/Research/1-BCNet/body_garment_dataset', './datasets', 'shirts')

if __name__ == '__main__':
    main_test()