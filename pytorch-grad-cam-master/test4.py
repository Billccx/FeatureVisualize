# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :explainmodel
# @File     :test4
# @Date     :2021/9/22 11:36
# @Author   :CuiChenxi
# @Email    :billcuichenxi@163.com
# @Software :PyCharm
-------------------------------------------------
"""
import torch
x=torch.rand(1,512,7,7)
x[:,0:0,:,:]=0.
x[:,1:,:,:]=0.
print(x[:,1,:,:])
