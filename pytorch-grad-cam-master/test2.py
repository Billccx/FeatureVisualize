# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :explainmodel
# @File     :test2
# @Date     :2021/9/22 0:22
# @Author   :CuiChenxi
# @Email    :billcuichenxi@163.com
# @Software :PyCharm
-------------------------------------------------
"""

import torch
import torch.nn as nn


def hook(module, grad_input, grad_output):
    print('grad_input: ', grad_input)
    print('grad_output: ', grad_output)
    return grad_input[0] * 0, grad_input[1] * 0, grad_input[2] * 0,

def hook2(module, grad_input, grad_output):
    print('grad_input: ', grad_input)
    print('grad_output: ', grad_output)
    return torch.tensor([[110.]],requires_grad=True)


x = torch.tensor([[1., 1., 1.],[2.,2.,2.]], requires_grad=True)
module = nn.Linear(3, 1)
handle = module.register_backward_hook(hook)
handle2= module.register_forward_hook(hook2)
y = module(x)
y.backward()
print('module_bias: ', module.bias.grad)
print('x: ', x.grad)
print('module_weight: ', module.weight.grad)

#handle.remove()
