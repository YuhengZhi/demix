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
from termcolor import colored
import numpy as np
import time
import functools

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

        self.apply(weights_init)

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


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal(m.weight.data)


def conv_norm_act(in_dim, out_dim, kernel_size, stride, padding=0,
                  norm=nn.BatchNorm2d, relu=nn.ReLU):
    return nn.Sequential(
                        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
                        norm(out_dim),
                        relu())


class Discriminator(nn.Module):

    def __init__(self, dim=64):
        super(Discriminator, self).__init__()

        lrelu = functools.partial(nn.LeakyReLU, negative_slope=0.2)
        conv_bn_lrelu = functools.partial(conv_norm_act, relu=lrelu)

        self.ls = nn.Sequential(nn.Conv2d(3, dim, 4, 2, 1), nn.LeakyReLU(0.2),
                                conv_bn_lrelu(dim * 1, dim * 2, 4, 2, 1),
                                conv_bn_lrelu(dim * 2, dim * 4, 4, 2, 1),
                                conv_bn_lrelu(dim * 4, dim * 8, 4, 1, (1, 2)),
                                nn.Conv2d(dim * 8, 1, 4, 1, (2, 1)))

        self.apply(weights_init)

    def forward(self, x):
        return self.ls(x)


# define Loss Function and Optimizer
class DemixLoss(nn.Module):
    def __init__(self, img_differ=0):
        super(DemixLoss, self).__init__()
        self.img_differ = img_differ

    def forward(self, x, demix_output):
        n_batch = x.size()[0]
        paired_batch_input = torch.cat((x[:n_batch // 2], x[n_batch // 2:]), dim=1)
        # alter_output = torch.cat((demix_output[:, 3:6], demix_output[:, 0:3]), dim=1)
        loss = nn.MSELoss()(demix_output, paired_batch_input)
        # alter_loss = nn.MSELoss()(alter_output, paired_batch_input)

        return loss
               # - self.img_differ * torch.mean((demix_output[:, 0:3]-demix_output[:, 3:6]) ** 2)  #torch.min(torch.cat((loss, alter_loss))) \

    def __call__(self, *args, **kwargs):
        return super(DemixLoss, self).__call__(*args, **kwargs)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def generate_mixup_ratio(alpha, min_mix=0.6, max_mix=0.9):
    return min_mix + (max_mix - min_mix) * np.random.beta(a=alpha, b=alpha)


if __name__ == '__main__':
    print(colored("DO NOT USE THIS CODE until you FIX the DATAPARRLLEL ERROR!"), 'red')
    exit(-1)

    parser = argparse.ArgumentParser(description='Pytorch DeMix Training')
    parser.add_argument('--data', metavar='PATH', default='data/tiny-imagenet-200',  # required=True,
                        help='path to dataset')
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
    #                     help='model architecture: resnet18 | resnet34 | ...'
    #                          '(default: alexnet)')
    parser.add_argument('--saved_model', default='saved_model', metavar='PATH',
                        help='path to save generated files (default: gen)')
    parser.add_argument('--logdir', default='log', metavar='PATH',
                        help='path to tensorboard events (default: \'log\'')
    parser.add_argument('--n_threads', '-j', default=8, type=int, metavar='N',
                        help='number of data loading threads (default: 8)')
    parser.add_argument('--n_epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run (default: 90)')
    parser.add_argument('--epoch_number', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', '-b', default=128, type=int, metavar='N',
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
    # parser.add_argument('--mix_ratio', default=0.8, type=float,
    #                     help='Ratio of the first image takes part in the mixed input (default: 0.8)')
    parser.add_argument('--mixup_alpha', default=-1, type=float,
                        help='Parameter of the beta distribution when choosing mixup rate at training')
    parser.add_argument('--mixup_min', default=0.6, type=float,
                        help='Minus mixup rate')
    parser.add_argument('--mixup_max', default=0.9, type=float,
                        help='Max mixup rate')
    parser.add_argument('--load_from_checkpoint', default='',
                        help='Path to restore saved model.')
    parser.add_argument('--continue_training', default=1, type=int,
                        help='Are you continuing previous training process?')
    parser.add_argument('--cuda', default=1, type=int,
                        help='Using cuda or cpu?')
    parser.add_argument('--mixup', default='input',
                        help='Choose where to mixup (input/feature): '
                             '[input, conv1, conv2, conv3, conv4, conv5]')
    parser.add_argument('--gpu_ids', default='0,1,2,3',
                        help='IDs of the GPUs you wish to use, comma-delimited. (default: 1,2,3,4)')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='L2 weight Decay rate')
    parser.add_argument('--d_lr', default=0.001, type=float,
                        help='Learning rate for discriminator')
    parser.add_argument('--adv_weight', default=0.001, type=float,
                        help='Weight for adversarial loss.')

    args = parser.parse_args()
    print(args)

    model = Demix(mix_begin=args.mixup)
    D = Discriminator(dim=64)
    if args.cuda:
        gpu_ids = [int(e) for e in args.gpu_ids.split(',')]
        print(colored(gpu_ids, 'red'))
        model = nn.DataParallel(model, device_ids=gpu_ids).cuda()  # , device_ids=gpu_ids
        D = nn.DataParallel(D, device_ids=gpu_ids).cuda()

    g_optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                  weight_decay=1e-5)  # torch.nn.DataParallel(, device_ids=gpu_ids) momentum=args.momentum)  # , args.momentum)
    d_optimizer = torch.optim.SGD(D.parameters(), args.d_lr, momentum=args.momentum,
                                  weight_decay=1e-5)

    criterion = DemixLoss(img_differ=args.img_differ)

    print("Testing:")
    x = torch.zeros([6, 3, 227, 227])
    x = Variable(x)
    if args.cuda:
        x = x.cuda()
    print(model.forward(x))

    cudnn.benchmark = True

    # Data loading code
    transform = transforms.Compose([
        transforms.RandomResizedCrop(227),
        # transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    train = datasets.ImageFolder(traindir, transform)
    print(train)
    # val = datasets.ImageFolder(valdir, transform)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=2 * args.batch_size, shuffle=True, num_workers=args.n_threads, drop_last=True)

    instance_logdir = os.path.join(args.logdir, 'mixup_'+args.mixup, str(args.lr))
    print("We are going to log at <", colored(instance_logdir, color='green'), '>')
    logger = Logger(instance_logdir)
    if not args.continue_training:
        for file in os.listdir(instance_logdir):
            if 'events' in file:
                os.remove(os.path.join(instance_logdir, file))

    if not args.load_from_checkpoint == '':
        checkpoint = torch.load(args.load_from_checkpoint)
        model.load_state_dict(checkpoint["G"])
        D.load_state_dict(checkpoint["D"])
        g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer"])
        args.epoch_number = checkpoint["epoch"]
        print("Restored model from {}".format(colored(args.load_from_checkpoint, 'green')))
        print("The args are {}".format(colored(checkpoint["args"], 'yellow')))

    model.train(mode=True)
    D.train(mode=True)

    for epoch in range(args.epoch_number, args.n_epochs):
        lr = args.lr * args.lr_decay ** (epoch)
        d_lr = args.d_lr * args.lr_decay ** (epoch)
        for param_group in g_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in d_optimizer.param_groups:
            param_group['lr'] = d_lr

        model_dir = '{}/mixup_{}/'.format(args.saved_model, args.mixup)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
            print(colored("Creating directory: " + model_dir, 'green'))
        torch.save({"epoch": epoch,
                    "G": model.state_dict(),
                    "D": D.state_dict(),
                    "g_optimizer": g_optimizer,
                    "d_optimizer": d_optimizer,
                    "args": args},
                   model_dir + 'lr_{}_adv_demix_epoch{}.pkl'.format(str(args.lr), epoch))

        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        for iteration, data in enumerate(train_loader, 1):
            data_time.update(time.time() - end)

            batch_input, _ = data

            batch_input = Variable(batch_input)
            if args.cuda:
                batch_input = batch_input.cuda()

            # G training
            mixup_ratio = generate_mixup_ratio(alpha=args.mixup_alpha, min_mix=args.mixup_min, max_mix=args.mixup_max)

            demix_output = model.forward(batch_input, mix_ratio=mixup_ratio)
            rec_batch = torch.cat((demix_output[:, :3], demix_output[:, 3:]), dim=0)
            f_dis = D(rec_batch)
            r_label = Variable(torch.ones(f_dis.size())).cuda()
            MSE = nn.MSELoss()
            gen_loss = MSE(f_dis, r_label)

            demix_loss = criterion(batch_input, demix_output)

            g_loss = demix_loss + args.adv_weight * gen_loss

            g_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            g_optimizer.step()

            # D training
            f_dis = D(rec_batch)
            r_dis = D(batch_input)
            f_label = Variable(torch.zeros(f_dis.size())).cuda()
            r_label = Variable(torch.ones(r_dis.size())).cuda()

            d_r_loss = MSE(r_dis, r_label)
            d_f_loss = MSE(f_dis, f_label)
            dis_loss = d_r_loss + d_f_loss

            d_loss = dis_loss * args.adv_weight

            D.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            log_interval = len(train_loader) // 50
            if iteration % log_interval == 0:
                print("Epoch {}, iteration {}/{}, lr: {}, d_lr: {}, g_loss: {}, demix_loss: {}, "
                      "gen_loss: {}, dis_loss: {}, data_time: {}, batch_time: {}"
                      .format(epoch, iteration, len(train_loader), lr, d_lr, g_loss.data[0],
                              demix_loss.data[0], gen_loss.data[0], dis_loss.data[0],
                              data_time.avg, batch_time.avg))
                info = {
                    'g_loss': g_loss.data[0],
                    'demix_loss': demix_loss.data[0],
                    'gen_loss': gen_loss.data[0],
                    'dis_loss': dis_loss.data[0],
                    'learning_rate': lr
                }
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step=iteration + (epoch-1) * len(train_loader))

            image_log_interval = len(train_loader) // 5
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

                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), iteration + epoch * len(train_loader))
                    logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), iteration + epoch * len(train_loader))

            batch_time.update(time.time() - end)
            end = time.time()


