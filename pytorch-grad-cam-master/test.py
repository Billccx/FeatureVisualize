# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :explainmodel
# @File     :test
# @Date     :2021/9/20 12:09
# @Author   :CuiChenxi
# @Email    :billcuichenxi@163.com
# @Software :PyCharm
-------------------------------------------------
"""
import torch
x=torch.ones(1,requires_grad=True)
z=x+1
y=z**3
t=y+2
m=t*6

m.backward(retain_graph=True)
print(m.grad)
print(t.grad)
print(y.grad)
print(z.grad)
print(x.grad)


x.grad=None
m.backward(retain_graph=True)
print(m.grad)
print(t.grad)
print(y.grad)
print(z.grad)
print(x.grad)

x.grad=None
m.backward(retain_graph=True)
print(m.grad)
print(t.grad)
print(y.grad)
print(z.grad)
print(x.grad)

x.grad=None
m.backward(retain_graph=True)
print(m.grad)
print(t.grad)
print(y.grad)
print(z.grad)
print(x.grad)

