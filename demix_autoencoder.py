import torch
import torch.nn as nn
from termcolor import colored


class Demix(nn.Module):
    def __init__(self, mix_begin="input"):
        super(Demix, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),  # 227 -> 55
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)  # 55 -> 27
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)  # 27 -> 13
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        layer_name = ['input', 'conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'conv5']
        layers = [None, self.conv1, self.pool1, self.conv2, self.pool2, self.conv3, self.conv4, self.conv5]

        independent_encoding_layers = []
        mixup_encoding_layers = []
        flag_independent = True
        for (name, layer) in zip(layer_name, layers):
            if layer is not None:
                if flag_independent:
                    independent_encoding_layers.append(layer)
                else:
                    mixup_encoding_layers.append(layer)
            if name == mix_begin:
                flag_independent = False
        if flag_independent:
            print(colored("Undefined mixup point!", color='Yellow'))
            exit(-1)

        print(independent_encoding_layers)
        print(mixup_encoding_layers)
        self.independent_encode = nn.Sequential(*independent_encoding_layers)
        self.mixup_encode = nn.Sequential(*mixup_encoding_layers)

        # self.encode = nn.Sequential([
        #     nn.MaxPool2d(kernel_size=3, stride=2),  # 55 -> 27
        #     nn.Conv2d(96, 256, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),  # 27 -> 13
        #     nn.Conv2d(256, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(kernel_size=3, stride=2)
        #     ]
        # )

        self.decode = nn.Sequential(
            # nn.Conv2d(256, 512, kernel_size=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # 13 -> 13
            nn.ConvTranspose2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 13 -> 27
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 27 -> 27
            nn.ConvTranspose2d(256, 192, 5, stride=1, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            # 27 -> 55
            nn.ConvTranspose2d(192, 192, 3, 2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            # 27 -> 227
            nn.ConvTranspose2d(192, 3, 11, 4),
            # nn.BatchNorm2d(6),
            nn.Tanh()
        )

    def forward(self, x, mix_ratio=0.8):
        n_batch = x.size()[0]
        # print(n_batch)

        pic1_interm = self.independent_encode(x[:n_batch // 2])
        pic2_interm = self.independent_encode(x[n_batch // 2:])

        mixup_interm = mix_ratio * pic1_interm + (1 - mix_ratio) * pic2_interm
        mixup_embedding = self.mixup_encode(mixup_interm)

        pic2 = self.decode(mixup_embedding)
        pic1 = (mixup_interm - (1 - mix_ratio) * pic2) / mix_ratio

        return torch.cat((pic1, pic2), dim=1)