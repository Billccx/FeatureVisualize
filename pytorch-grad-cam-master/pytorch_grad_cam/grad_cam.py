import cv2
import numpy as np
import torch
from pytorch_grad_cam.base_cam import BaseCAM

class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False, 
        reshape_transform=None):
        # 首先找到GradCAM的父类BaseCAM，然后把类GradCAM的对象self转换为类BaseCAM的对象，然后“被转换”的类BaseCAM对象调用自己的_init_函数
        # 这是对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
        super(GradCAM, self).__init__(model, target_layer, use_cuda, reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))