import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.trainer as trainer
import torch.utils.trainer.plugins
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from logger import Logger

# class Args:
#     def __init__(self, arg_dict):
#         self.data = arg_dict['data']
#         self.arch = arg_dict['arch']
#         self.nThreads = arg_dict['nThreads']
#         self.nEpochs = arg_dict['nEpochs']
#         self.epochNumber = arg_dict['epochNumber']
#         self.batchSize = arg_dict['batchSize']
#         self.lr = arg_dict['lr']
#         self.momentum = arg_dict['momentum']
#         self.weightDecay = arg_dict['weightDecay']
#         self.cuda = arg_dict['cuda']
#         self.checkpoint = arg_dict['checkpoint']
#         self.lr_decay = arg_dict['lr_decay']
#         self.img_differ = arg_dict['img_differ']
#         self.mix_ratio = arg_dict['mix_ratio']


class Demix(nn.Module):
    def __init__(self, mix_begin="input"):
        super(Demix, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # 227 -> 55
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2),  # 55 -> 27
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2),  # 27 -> 13
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        layer_name = ['input', 'conv1', 'pool1', 'conv2', 'pool2',
                      'conv3', 'conv4', 'conv5']
        layers = [None, self.conv1, self.pool1, self.conv2, self.pool2,
                  self.conv3, self.conv4, self.conv5]

        independent_encoding_layers = []
        mixup_encoding_layers = []

        flag_independent = True
        for name, layer in zip(layer_name, layers):
            if flag_independent and layer is not None:
                independent_encoding_layers.append(layer)
            else:
                mixup_encoding_layers.append(layer)
            if name == mix_begin:
                flag_independent = False

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
            nn.Conv2d(256, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 768, kernel_size=3, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(768, 768, kernel_size=3, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),

            # 13 -> 13
            nn.ConvTranspose2d(768, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 13 -> 27
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 27 -> 27
            nn.ConvTranspose2d(512, 384, 5, stride=1, padding=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # 27 -> 55
            nn.ConvTranspose2d(384, 384, 3, 2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # 27 -> 227
            nn.ConvTranspose2d(384, 6, 11, 4),
            # nn.BatchNorm2d(6),
            nn.Tanh()
        )

    def forward(self, x, mix_ratio):
        n_batch = x.size()[0]
        # print(n_batch)

        pic1_interm = self.independent_encode(x[:n_batch // 2])
        pic2_interm = self.independent_encode(x[n_batch // 2:])

        mixup_interm = self.mix_ratio * pic1_interm + (1-self.mix_ratio) * pic2_interm
        mixup_embedding = self.mixup_encode(mixup_interm)

        reconst = self.decode(mixup_embedding)

        return reconst


# define Loss Function and Optimizer
class DemixLoss(nn.Module):
    def __init__(self, img_differ=0):
        super(DemixLoss, self).__init__()
        self.img_differ = img_differ

    def forward(self, x, demix_output):
        n_batch = batch_input.size()[0]
        paired_batch_input = torch.cat((batch_input[:n_batch // 2], batch_input[n_batch // 2:]), dim=1)
        alter_output = torch.cat((demix_output[:, 3:6], demix_output[:, 0:3]), dim=1)
        loss = nn.MSELoss()(demix_output, paired_batch_input)
        alter_loss = nn.MSELoss()(alter_output, paired_batch_input)

        return loss \
               - self.img_differ * torch.mean((demix_output[:, 0:3]-demix_output[:, 3:6]) ** 2)  #torch.min(torch.cat((loss, alter_loss))) \

    def __call__(self, *args, **kwargs):
        return super(DemixLoss, self).__call__(*args, **kwargs)


if __name__ == '__main__':

    # args = Args({
    #     'data': 'data/tiny-imagenet-200',
    #     'arch': 'alexnet',
    #     'gen': 'gen',
    #     'nThreads': 8,
    #     'nEpochs': 90,
    #     'epochNumber': 1,
    #     'batchSize': 128,
    #     'lr': 0.1,
    #     'momentum': 0.9,
    #     'weightDecay': 1e-4,
    #     'cuda': True,
    #     'checkpoint': 'saved_model',
    #     'lr_decay': 0.99,
    #     'img_differ': 0,
    #     'mix_ratio': 0.7  # Ensure this is not 0.5
    # })

    parser = argparse.ArgumentParser(description='Pytorch DeMix Training')
    parser.add_argument('--data', metavar='PATH', default='data/tiny-imagenet-200',  # required=True,
                        help='path to dataset')
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
    #                     help='model architecture: resnet18 | resnet34 | ...'
    #                          '(default: alexnet)')
    parser.add_argument('--saved_model', default='saved_model', metavar='PATH',
                        help='path to save generated files (default: gen)')
    parser.add_argument('--tensorboard', default='log', metavar='PATH',
                        help='path to tensorboard events (default: \'log\'')
    parser.add_argument('--nThreads', '-j', default=8, type=int, metavar='N',
                        help='number of data loading threads (default: 8)')
    parser.add_argument('--nEpochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run (default: 90)')
    parser.add_argument('--epochNumber', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batchSize', '-b', default=128, type=int, metavar='N',
                        help='mini-batch size (1 = pure stochastic) Default: 128')
    parser.add_argument('--lr', default=0.01, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--lr_decay', default=0.99, type=float, metavar='LR_DECAY',
                        help='Learning rate = LR*LR_DECAY^EPOCH (default: 0.99)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # parser.add_argument('--weightDecay', default=1e-4, type=float, metavar='W',
    #                     help='weight decay')
    parser.add_argument('--img_differ', default=0, type=float, help='Weight of pushing two outputs to be different')
    parser.add_argument('--mix_ratio', default=0.8, type=float,
                        help='Ratio of the first image takes part in the mixed input (default: 0.8)')
    parser.add_argument('--load_from_checkpoint', default='',
                        help='Path to restore saved model.')
    parser.add_argument('--continue_training', default=1, type=int,
                        help='Are you continuing previous training process?')
    parser.add_argument('--cuda', default=1, type=int,
                        help='Using cuda or cpu?')

    args = parser.parse_args()
    print(args)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print(args.__dict__)

    model = Demix(mix_ratio=args.mix_ratio)
    if args.cuda:
        model = model.cuda()

    print("Testing:")
    x = torch.zeros([2, 3, 227, 227])
    x = Variable(x)
    if args.cuda:
        x = x.cuda()
    print(model.forward(x))

    cudnn.benchmark = True

    # Data loading code
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    train = datasets.ImageFolder(traindir, transform)
    print(train)
    # val = datasets.ImageFolder(valdir, transform)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=2 * args.batchSize, shuffle=True, num_workers=args.nThreads)
    # train_loader.dataset.labels.cuda()

    criterion = DemixLoss(img_differ=args.img_differ)

    logger = Logger("./log")
    if not args.continue_training:
        for file in os.listdir("./log"):
            if 'events' in file:
                os.remove('./log/'+file)

    if not args.load_from_checkpoint == '':
        checkpoint = torch.load(args.load_from_checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        args.epochNumber = checkpoint["epoch"]
        print("Restored model from {}".format(args.load_from_checkpoint))
        print("The args are {}".format(checkpoint["args"]))

    for epoch in range(args.epochNumber, args.nEpochs):
        lr = args.lr * args.lr_decay ** (epoch)
        optimizer = torch.optim.Adam(model.parameters(), lr)  # momentum=args.momentum)  # , args.momentum)
        for iteration, data in enumerate(train_loader, 1):
            optimizer.zero_grad()
            batch_input, _ = data

            batch_input = Variable(batch_input)
            if args.cuda:
                batch_input = batch_input.cuda()

            demix_output = model.forward(batch_input)
            # alternative_output = torch.cat((output[:, 3:6], output[:, 0:3]), dim=1)
            demix_loss = criterion(batch_input, demix_output)

            demix_loss.backward()

            optimizer.step()

            log_interval = len(train_loader) // 50
            if iteration % log_interval == 0:
                print("Epoch {}, iteration {}/{}, lr: {}, demix_loss: {}".format(epoch, iteration, len(train_loader), lr, demix_loss.data[0]))
                info = {
                    'loss': demix_loss.data[0],
                    'learning_rate': lr
                }
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step=iteration + epoch * len(train_loader))

            image_log_interval = len(train_loader) // 2
            if iteration % image_log_interval == 0:
                n_batch = batch_input.size()[0]
                print(n_batch)
                original_image_1 = batch_input[:10]
                original_image_2 = batch_input[(n_batch//2):(n_batch//2 + 10)]

                def de_normalize(batch_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
                    de_transform = transforms.Compose([
                        transforms.Normalize(mean=[0., 0., 0.],
                                             std=[1. / ch_std for ch_std in std]),
                        transforms.Normalize(mean=[- ch_mean for ch_mean in mean],
                                             std=[1., 1., 1.])
                    ])
                    for i in range(batch_image.size()[0]):
                        batch_image[i] = de_transform(batch_image[i])
                    return batch_image

                original_image_1 = de_normalize(original_image_1.data).cpu().numpy()
                original_image_2 = de_normalize(original_image_2.data).cpu().numpy()
                info = {
                    'Original Image 1': original_image_1,
                    'Original Image 2': original_image_2,
                    'Mixed Image': (original_image_1 + original_image_2)/2,
                    'Reconstructed Image 1': de_normalize(demix_output[:10, 0:3].data).cpu().numpy(),
                    'Reconstructed Image 2': de_normalize(demix_output[:10, 3:6].data).cpu().numpy()
                }
                for tag, images in info.items():
                    logger.image_summary(tag, images, step=iteration + epoch * len(train_loader))

        torch.save({"epoch": epoch,
                    "state_dict":model.state_dict(),
                    "args": args},
                   '{}/demix_epoch{}.pkl'.format(args.saved_model, epoch))
