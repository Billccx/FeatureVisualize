# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :explainmodel
# @File     :main
# @Date     :2021/9/22 12:08
# @Author   :CuiChenxi
# @Email    :billcuichenxi@163.com
# @Software :PyCharm
-------------------------------------------------
"""
import argparse
import cv2
import numpy as np
import torch
import torchvision
from torchvision import models
from torch import nn
import time
import os
import sys

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 'eigengradcam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

if __name__ == '__main__':
    print("Torchvision Version: ", torchvision.__version__)
    print("torch Version: ", torch.__version__)
    Normalize_info_dict2 = {
        "Nairobi": [[0.43289095, 0.46085805, 0.4772808], [0.24355671, 0.25768152, 0.28018752]],
        "CapeTown": [[0.38198748, 0.41382942, 0.44994584], [0.19300567, 0.21488968, 0.2430879]],
    }
    class_num = 2
    country_name = 'Nairobi'
    model_name = 'Nairobi_resnet18_mapillary_2_3.2'  # 模型名称  Mumbai_resnet18_mapillary_2_1.1
    model_country = model_name.split('_')[0]
    layer_name = 'layer4'
    my_method = 'gradcam'
    my_aug_smooth = False
    my_eigen_smooth = False

    data_dir = '..\\img_data'
    cam_dest_folder = '..\\result\\cam'
    gb_dest_folder = '..\\result\\gb'
    cam_gb_dest_folder = '..\\result\\camgb'
    logpath = '..\\log'
    model_dir = '..\\model/Nairobi_resnet18_mapillary_2_3.2.pt'

    if not os.path.exists(cam_dest_folder):
        os.makedirs(cam_dest_folder)
    if not os.path.exists(gb_dest_folder):
        os.makedirs(gb_dest_folder)
    if not os.path.exists(cam_gb_dest_folder):
        os.makedirs(cam_gb_dest_folder)
    input_size = 224
    if torch.cuda.is_available():
        print("cuda:0")
        my_use_cuda = True
    else:
        print("cpu")
        my_use_cuda = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_dir)  # 可以改为model_ft = models.resnet18(pretrained=True)，直接下载ResNet18.
    model.eval()

    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM
    }
    target_layer = model.layer4[-1]

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=my_use_cuda)

    try:
        num = 0
        for img_set in ['train']:
            source_folder = os.path.join(data_dir, img_set)
            #请将需要做可视化的图片放在img_data/train/slum_test路径下
            #可视化结果位于result/gb/train/slum_test路径下
            for label in ['slum_test']:
                img_folder = os.path.join(source_folder, label)
                cam_df = os.path.join(cam_dest_folder, img_set, label)
                gb_df = os.path.join(gb_dest_folder, img_set, label)
                cam_gb_df = os.path.join(cam_gb_dest_folder, img_set, label)
                if not os.path.exists(cam_df):
                    os.makedirs(cam_df)
                if not os.path.exists(gb_df):
                    os.makedirs(gb_df)
                if not os.path.exists(cam_gb_df):
                    os.makedirs(cam_gb_df)
                done_img_list = []


                for img_name in os.listdir(img_folder):
                    if img_name not in done_img_list:
                        img_path = os.path.join(img_folder, img_name)

                        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
                        rgb_img = cv2.resize(rgb_img, (224, 224))
                        rgb_img = np.float32(rgb_img) / 255
                        input_tensor = preprocess_image(rgb_img, mean=Normalize_info_dict2[model_country][0],
                                                        std=Normalize_info_dict2[model_country][1])
                        input_tensor = input_tensor.to(device)
                        target_category = None

                        for i in range(512):
                            print(i)
                            gb = gb_model(input_tensor, target_category=target_category)

                            gb = deprocess_image(gb)
                            tarfold='../result/gb/train/slum_test/'+img_name.split('.')[0]
                            tarpath='../result/gb/train/slum_test/'+img_name.split('.')[0]+'/map'+str(i)+'.jpg'
                            if not os.path.exists(tarfold):
                                os.makedirs(tarfold)
                            cv2.imwrite(tarpath, gb)


                        gb_model.resetCnt()

                        #print(num)
                        num += 1

                        '''
                        print("after remove hook:")
                        gb_model.hd.remove()
                        gb = gb_model(input_tensor, target_category=target_category)
                        gb = deprocess_image(gb)
                        print(gb)
                        '''

                    if num % 100 == 0:
                        print('---do %s img---' % num)
                print('---------%s,%s is OK---------' % (img_set, label))


    except Exception as e:
        print('*************************************')
        print(e)
        print('wrong img is : %s-%s-%s' % (img_set, label, img_name))
        print('*************************************')

