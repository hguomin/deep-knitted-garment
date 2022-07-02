from __future__ import annotations
import os
import pickle
import torch
import numpy as np
import open3d as o3d
from metro.datasets.human_mesh_tsv import MeshTSVYamlDataset
from metro.modeling._smpl import SMPL
import metro.modeling.data.config as data_config

def get_dataset_loader(data_dir, data_meta_file, items_per_batch=1, is_train=False, scale_factor=1):
    if not os.path.isfile(data_meta_file):
        data_meta_file = os.path.join(data_dir, data_meta_file)
        assert os.path.isfile(data_meta_file)

    dataset = MeshTSVYamlDataset(data_meta_file, is_train, False, scale_factor)
    sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, items_per_batch, drop_last=False)

    return torch.utils.data.DataLoader(dataset, num_workers=3, batch_sampler=batch_sampler, pin_memory=True)

def main(opts):
    dataloader = get_dataset_loader("datasets", "3dpw/test_has_gender.yaml")
    smpl = SMPL().to(opts["device"])
    sim_data = dict()
    for iteration, (img_keys, images, annotations) in enumerate(dataloader):
        print(img_keys)
        gt_pose = torch.zeros([1, 72]).cuda(opts["device"]) #annotations["pose"].cuda(opts["device"])
        gt_betas = annotations["betas"].cuda(opts["device"])
        gt_vertices = smpl(gt_pose, gt_betas)
        gt_faces = smpl.faces
        # Normalize gt_vertices based on smpl's pelvis
        gt_smpl_3d_joints = smpl.get_h36m_joints(gt_vertices)
        gt_smpl_3d_pelvis = gt_smpl_3d_joints[:,data_config.H36M_J17_NAME.index("Pelvis"),:]
        gt_vertices = gt_vertices - gt_smpl_3d_pelvis[:, None, :]

        np_vertices = gt_vertices[0].cpu().numpy()
        np_faces = gt_faces.cpu().numpy()

        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np_vertices), o3d.utility.Vector3iVector(np_faces.astype(np.int32)))
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(filename="output/xxx.obj", mesh=mesh)

        img_data = dict()
        img_data['pose'] = gt_pose[0].cpu().numpy()
        img_data["betas"] = gt_betas[0].cpu().numpy()
        img_data["vertices"] = np_vertices
        img_data["faces"] = np_faces
        sim_data[img_keys[0]] = img_data
        file = open("output/datas.pkl", "wb")
        pickle.dump(sim_data, file)
        file.close()

        file2 = open("output/datas.pkl", "rb")
        mesh2 = pickle.load(file2)
        file2.close()
        #o3d.visualization.draw_geometries([mesh])
    return

if __name__ == "__main__":
    file2 = open("output/datas.pkl", "rb")
    mesh2 = pickle.load(file2)
    file2.close()
    opts = {}
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    
    opts["device"] = torch.device(device)

    main(opts)