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
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """
    print("Torchvision Version: ",torchvision.__version__)
    print("torch Version: ",torch.__version__)
    Normalize_info_dict2 = {
        "Nairobi":[[0.43289095,0.46085805,0.4772808],[0.24355671,0.25768152,0.28018752]],
        "CapeTown":[[0.38198748,0.41382942,0.44994584],[0.19300567,0.21488968,0.2430879]],
    }
    class_num = 2
    country_name = 'Nairobi'
    model_name = 'Nairobi_resnet18_mapillary_2_3.2'# 模型名称  Mumbai_resnet18_mapillary_2_1.1
    model_country = model_name.split('_')[0]
    layer_name = 'layer4'
    my_method = 'gradcam'
    my_aug_smooth = False
    my_eigen_smooth = False

    # data_dir = os.path.join('/workspace/pytorch_lm/imgdata/',country_name,'img3.2_'+country_name)# Mapillary照片数据路径
    # cam_dest_folder = os.path.join('/workspace/pytorch_lm/visualization/',country_name,'cam_'+layer_name)
    # gb_dest_folder = os.path.join('/workspace/pytorch_lm/visualization/',country_name,'gb_'+layer_name)
    # cam_gb_dest_folder = os.path.join('/workspace/pytorch_lm/visualization/',country_name,'cam_gb_'+layer_name)
    # logpath = os.path.join('/workspace/pytorch_lm/visualization/',country_name)
    # model_dir = os.path.join('/workspace/pytorch_lm/model/',model_name+'.pt')# 模型路径

    '''
    data_dir = 'D:\\geopicture\\feature_extraction\\visualization_test_data' # Mapillary照片数据路径
    cam_dest_folder = os.path.join('D:\\geopicture\\feature_extraction\\',country_name,'cam_'+layer_name)
    gb_dest_folder = os.path.join('D:\\geopicture\\feature_extraction\\',country_name,'gb_'+layer_name)
    cam_gb_dest_folder = os.path.join('D:\\geopicture\\feature_extraction\\',country_name,'cam_gb_'+layer_name)
    logpath = os.path.join('D:\\geopicture\\feature_extraction\\',country_name)
    model_dir = os.path.join('D:\\geopicture\\model\\',model_name+'.pt')# 模型路径
    '''
    data_dir='..\\img_data'
    cam_dest_folder ='..\\result\\cam'
    gb_dest_folder='..\\result/gb'
    cam_gb_dest_folder ='..\\result\\camgb'
    logpath='..\\log'
    model_dir='..\\model/Nairobi_resnet18_mapillary_2_3.2.pt'

    if not os.path.exists(cam_dest_folder):
        os.makedirs(cam_dest_folder)
    if not os.path.exists(gb_dest_folder):
        os.makedirs(gb_dest_folder)
    if not os.path.exists(cam_gb_dest_folder):
        os.makedirs(cam_gb_dest_folder)
    input_size = 224
    if torch.cuda.is_available():
        print ("cuda:0")
        my_use_cuda = True
    else:
        print("cpu")
        my_use_cuda = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_dir)#可以改为model_ft = models.resnet18(pretrained=True)，直接下载ResNet18.
    model.eval()
    print (model)
    # print ('===================================')

    # args = get_args()
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
    # print (target_layer)
    # print ('=====================================')
    # print (model.layer4)
    
    cam = methods[my_method](model=model,target_layer=target_layer,use_cuda=my_use_cuda)
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=my_use_cuda)

    try:
        num = 0
        # for img_set in os.listdir(data_dir):
        for img_set in ['train']:
            source_folder = os.path.join(data_dir,img_set)
            # for label in os.listdir(source_folder):
            for label in ['slum']:
                img_folder = os.path.join(source_folder,label)
                cam_df = os.path.join(cam_dest_folder,img_set,label)
                gb_df = os.path.join(gb_dest_folder,img_set,label)
                cam_gb_df = os.path.join(cam_gb_dest_folder,img_set,label)
                if not os.path.exists(cam_df):
                    os.makedirs(cam_df)
                if not os.path.exists(gb_df):
                    os.makedirs(gb_df)
                if not os.path.exists(cam_gb_df):
                    os.makedirs(cam_gb_df)
                done_img_list = []
                # for done_img in os.listdir(cam_df):
                #     done_img_list.append(done_img[4:])
                for img_name in os.listdir(img_folder):
                    if img_name not in done_img_list:
                        img_path = os.path.join(img_folder,img_name)
                        cam = methods[my_method](model=model,target_layer=target_layer,use_cuda=my_use_cuda)
                        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
                        rgb_img = cv2.resize(rgb_img, (224, 224))
                        rgb_img = np.float32(rgb_img) / 255
                        input_tensor = preprocess_image(rgb_img, mean=Normalize_info_dict2[model_country][0], 
                                                                 std=Normalize_info_dict2[model_country][1])
                        input_tensor = input_tensor.to(device)
                        target_category = None
                        cam.batch_size = 32
                        grayscale_cam = cam(input_tensor=input_tensor,
                                            target_category=target_category,
                                            aug_smooth=my_aug_smooth,
                                            eigen_smooth=my_eigen_smooth)
                        grayscale_cam = grayscale_cam[0, :]
                        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

                        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=my_use_cuda) #导向反向传播(Guided-backpropagation)
                        gb = gb_model(input_tensor, target_category=target_category)

                        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
                        cam_gb = deprocess_image(cam_mask * gb)
                        gb = deprocess_image(gb)

                        cv2.imwrite(os.path.join(cam_df,'cam_'+img_name), cam_image)
                        cv2.imwrite(os.path.join(gb_df,'gb_'+img_name), gb)
                        cv2.imwrite(os.path.join(cam_gb_df,'cam_gb_'+img_name), cam_gb)
                        num += 1
                    if num%100 == 0:
                        print ('---do %s img---'%num)
                print ('---------%s,%s is OK---------'%(img_set,label))
    except Exception as e:
        print ('*************************************')
        print (e)
        print ('wrong img is : %s-%s-%s'%(img_set,label,img_name))
        print ('*************************************')
