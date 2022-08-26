import os
import os.path as osp
import numpy as np
import torch
from metro.modeling._smpl import SMPL

from psbody.mesh import Mesh, MeshViewers
import mesh_operations

def main():
    # change working directory to metro
    this_dir = osp.dirname(__file__)
    metro_dir = osp.join(this_dir, '..', 'libs', 'MeshTransformer')
    cwd = os.getcwd()
    os.chdir(metro_dir)
    smpl = SMPL().cuda()
    os.chdir(cwd)

    # Generate T-pose template mesh
    template_pose = torch.zeros((1,72))

    # Rectify "upside down" reference mesh in global coord
    template_pose[:,0] = 3.1416 
    template_pose = template_pose.cuda()
    template_betas = torch.zeros((1,10)).cuda()
    template_vertices = smpl(template_pose, template_betas)

    vertices = np.asarray(template_vertices[0].cpu())
    faces = np.asarray(smpl.faces.cpu())

    mesh = Mesh(v=vertices, f=faces)

    M, A, D, U = mesh_operations.generate_transform_matrices(mesh=mesh, factors=[2,4,8,16])
    return

if __name__ == "__main__":
    main()
