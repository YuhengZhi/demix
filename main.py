import torchvision.models as models
import torch.nn as nn
import torch
from torch.autograd import Variable

models.alexnet()
c = nn.ConvTranspose2d(1, 1, 5)
print(c)
input_feature = torch.Tensor([[2, 2], [2, 2]])
input_feature = Variable(input_feature)
input_feature = input_feature[None, None, :, :]
print(c(input_feature))

nn.Container


