import open3d as o3d

mesh = o3d.geometry.TriangleMesh.create_sphere()
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh("C:\\Users\\guhuang\\Desktop\\sphere.obj", mesh)
mesh2 = o3d.io.read_triangle_mesh("C:\\Users\\guhuang\\Desktop\\beixin.obj")
mesh2.compute_vertex_normals()
mesh2.translate([0,-1500.0,0])
o3d.io.write_triangle_mesh("C:\\Users\\guhuang\\Desktop\\beixin2.obj", mesh2)
#o3d.visualization.draw([mesh, mesh2], raw_mode=True)
