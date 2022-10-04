import numpy as np
import trimesh

if __name__ == '__main__':
    mesh = trimesh.load('/media/guomin/Works/Projects/Research/1-BCNet/recs1/female-4-sport_0_up.obj')
    rotateX = trimesh.transformations.rotation_matrix(np.pi, [1,0,0])
    mesh.apply_transform(rotateX)
    scene = trimesh.Scene({'mesh': mesh})
    print(scene.graph.nodes)
    scene.show()