#Guomin @2022.09.11
import sys
import os.path as osp
def add_sys_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

#In order to use openpose to estimation poses in an image, please build it from source first: https://blog.csdn.net/weixin_44379605/article/details/121980761 
#and set the openpose repo folder path here
openpose_repo_path = '/media/guomin/Works/Projects/Research/openpose'
openpose_models_path = osp.join(openpose_repo_path, 'models')
openpose_lib_path = osp.join(openpose_repo_path, 'build/python')

add_sys_path(openpose_lib_path)

import cv2
from openpose import pyopenpose as op

# Custom Params (refer to openpose's include/openpose/flags.hpp for more parameters and
# openpose's src/openpose/pose/poseParameters.cpp for different model_pose type and its joints definition, POSE_COCO_BODY_PARTS etc.)
params = dict()
params["model_folder"] = openpose_models_path
params["model_pose"] = 'COCO' #'COCO', 'MPI', 'BODY_25'
# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

def estimate_2d_joints(img_file):
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(img_file)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    return datum.poseKeypoints, datum.cvOutputData

