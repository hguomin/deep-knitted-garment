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

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = openpose_models_path

    # Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Process Image
datum = op.Datum()
imageToProcess = cv2.imread('/media/guomin/Works/Projects/Research/1-BCNet/body_garment_dataset/motion_datas/all_train_render_datas/SPRING0012_shirts_3_pants_3_28_07/vmode1/2980.jpg')
datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # Display Image
print("Body keypoints: \n" + str(datum.poseKeypoints))
cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
cv2.waitKey(0)
