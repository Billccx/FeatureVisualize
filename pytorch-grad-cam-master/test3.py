# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :explainmodel
# @File     :test3
# @Date     :2021/9/22 9:31
# @Author   :CuiChenxi
# @Email    :billcuichenxi@163.com
# @Software :PyCharm
-------------------------------------------------
"""
import torch
import torch.nn as nn
import numpy as np

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc0=nn.Linear(3,3)
        self.fc=nn.Linear(3,1)
        self.hd=self.fc.register_forward_hook(self.hook2)
        #self.fc.register_backward_hook(self.hook)
        self.cnt=0

    def increase(self):
        self.cnt+=1

    def forward(self,x):
        x=self.fc0(x)
        x=self.fc(x)
        y=torch.sum(x)
        return y

    #反向hook
    def hook(self, module, grad_input, grad_output):
        print('grad_input: ', grad_input)
        print('grad_output: ', grad_output)
        ret = grad_input.clone()
        ret[self.cnt][0] = 0.
        return ret

    def hook2(self, module, grad_input, grad_output):
        #print('grad_input: ', grad_input)
        #print('grad_output: ', grad_output)
        ret = grad_output.clone()
        ret[self.cnt][0] = 0.
        return ret


x = torch.tensor([[1., 1., 1.],[2.,2.,2.]], requires_grad=True)
model=net()
for i in range(2):
    x.grad=None
    y = model.forward(x)
    y.backward()
    print(x.grad)
    model.increase()

model.hd.remove()
x.grad=None
y = model.forward(x)
y.backward()
print(x.grad)


