import torch
from torch.autograd import Variable

#生成一个内容为[2,3]的张量，Varibale 默认是不要求梯度的，如果要求梯度，
#需要加上requires_grad=True来说明
a = Variable(torch.Tensor([[4,5],[1,2]]),requires_grad=True)
w = Variable(torch.Tensor([[2],[3]]),requires_grad=True)
out = torch.mm(a,w)

#括号里面的参数要传入和out维度一样的矩阵
#这个矩阵里面的元素会作为最后加权输出的权重系数
out.backward(torch.FloatTensor([[1],[1]]))
print (a)
print (w)
print (out)
print("gradients are:{}".format(w.grad.data))