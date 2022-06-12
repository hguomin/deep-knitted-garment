import os
import bpy

#proj_dir = "D:\\Projects\\DZJ\\deep-knitted-garment"
garmentTemplate = os.path.join("D:\\Projects\\DZJ\\deep-knitted-garment\\datasets\\templates\\garment\\beixin\\beixin.obj")

#print(garmentTemplate)

bpy.ops.import_scene.obj(filepath=garmentTemplate)

bpy.data.objects['beixin'].scale = [0.01, 0.01, 0.01]