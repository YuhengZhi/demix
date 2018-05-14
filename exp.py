import demix
import torch
from torch.autograd import Variable


class ID(torch.nn.Module):
    def __init__(self):
        super(ID, self).__init__()

    def forward(self, input):
        return input


torch.manual_seed(seed=1)

input = Variable(torch.zeros(256, 3, 227, 227).cuda())
# target = Variable(torch.ones(8, 3, 227, 227).cuda())
model = demix.Demix(mix_begin='input').cuda()
loss = demix.DemixLoss()
print(model(input))
print(loss(input, model.forward(input)))
print(loss(input[:4], model.forward(input[:4])))

dp = torch.nn.DataParallel(model, [0, 1])
print(loss(input, dp(input)))
print(loss(input[:4], dp(input[:4])))
