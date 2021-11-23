import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import datasets
import models as models
import matplotlib.pyplot as plt
import torchvision.models as torch_models
from extra_setting import *
import scipy.io as sio
import torchvision.utils as vutils
from models import cfNet
import matplotlib
import cv2


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch counterfactual GAN')
parser.add_argument('-d', '--dataset', default='capital_letter', help='dataset name')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-c', '--channel', type=int, default=16,
                    help='first conv channel (default: 16)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--gpu', default='0,3', help='index of gpus to use')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr_step', default='8000', help='decreasing strategy')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--first_epochs', default=80, type=int, metavar='N',
                    help='number of first stage epochs to run')
parser.add_argument('--lambda1', type=float, default=1.0, help='weight in loss function. default=1.0')
parser.add_argument('--lambda2', type=float, default=1.0, help='weight in loss function. default=1.0')
parser.add_argument('--lambda3', type=float, default=1.0, help='weight in loss function. default=1.0')
parser.add_argument('--lambda4', type=float, default=1.0, help='weight in loss function. default=1.0')


# somehow perfect results, solve the problem in cfGAN2, the model know where to add and where to erase


def main():
    global args
    args = parser.parse_args()

    # training multiple times

    # select gpus
    args.gpu = args.gpu.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    # data loader
    assert callable(datasets.__dict__[args.dataset])
    get_dataset = getattr(datasets, args.dataset)
    num_classes = datasets._NUM_CLASSES[args.dataset]
    train_loader, val_loader = get_dataset(
        batch_size=args.batch_size, num_workers=args.workers)




    # create model
    model_netG = cfNet.Generator(c_dim=26)
    model_netD = cfNet.Discriminator(c_dim=26)
    model_clser = models.__dict__['resnet20'](16, 26)

    model_netG = torch.nn.DataParallel(model_netG, device_ids=range(len(args.gpu))).cuda()
    model_netD = torch.nn.DataParallel(model_netD, device_ids=range(len(args.gpu))).cuda()
    model_clser = torch.nn.DataParallel(model_clser, device_ids=range(len(args.gpu))).cuda()

    if os.path.isfile('./capital_letter/checkpoint_pretrain_res20_28by28.pth.tar'):
        print("=> loading checkpoint '{}'".format('./capital_letter/checkpoint_pretrain_res20_28by28.pth.tar'))
        checkpoint = torch.load('./capital_letter/checkpoint_pretrain_res20_28by28.pth.tar')
        model_clser.module.load_state_dict(checkpoint['state_dict_m'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format('./capital_letter/checkpoint_pretrain_res20_28by28.pth.tar', checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format('./capital_letter/checkpoint_pretrain_res20_28by28.pth.tar'))

    criterion_kldiv = nn.KLDivLoss().cuda()
    criterion_CEL = nn.CrossEntropyLoss().cuda()

    # setup optimizer
    optimizerG = torch.optim.Adam(model_netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(model_netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

    cudnn.benchmark = True

    lr_step = list(map(int, args.lr_step.split(',')))
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in lr_step:
            for param_group in optimizerG.param_groups:
                param_group['lr'] *= 0.1
            for param_group in optimizerD.param_groups:
                param_group['lr'] *= 0.1
        train(train_loader, model_netG, model_netD, model_clser, optimizerG, optimizerD, epoch, criterion_CEL)


    # save_checkpoint({
    #     'epoch': args.epochs,
    #     'arch': args.arch,
    #     'state_dict_g': model_netG.state_dict(),
    #     'state_dict_d': model_netD.state_dict(),
    #     'optimizerG': optimizerG.state_dict(),
    #     'optimizerD': optimizerD.state_dict(),
    # }, filename='./mnist/checkpoint_res3.pth.tar')





def train(train_loader, model_netG, model_netD, model_clser, optimizerG, optimizerD, epoch, criterion_CEL):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_g = AverageMeter()
    losses_g_fake = AverageMeter()
    losses_g_reconst = AverageMeter()
    losses_g_cls = AverageMeter()
    losses_d = AverageMeter()
    losses_d_real = AverageMeter()
    losses_d_fake = AverageMeter()
    losses_d_cls = AverageMeter()

    top1 = AverageMeter()

    # switch to train mode
    # model_netG.train()
    # model_netD.train()
    model_clser.eval()

    end = time.time()
    for i, (input, target, index) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #

        x_real = input.cuda()
        label_org = target.cuda(async=True)

        # Generate target domain labels randomly.
        rand_idx = torch.randperm(label_org.size(0))
        label_trg = label_org[rand_idx]
        c_org = label2onehot(label_org, 26)
        c_trg = label2onehot(label_trg, 26)

        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #

        # Compute loss with real images.
        out_src, out_cls = model_netD(x_real)  # the prob of classifying real image to real image, we hope it large
        d_loss_real = - torch.mean(out_src)
        losses_d_real.update(d_loss_real.item(), input.size(0))
        d_loss_cls = criterion_CEL(out_cls, label_org)
        losses_d_cls.update(d_loss_cls.item(), input.size(0))

        # Compute loss with fake images.
        x_fake_delta = model_netG(x_real, label_trg)
        x_fake = x_real + 2.0 * x_fake_delta


        x_fake = torch.clamp(x_fake, -1.0, 1.0)

        out_src, out_cls = model_netD(x_fake.detach())  # the prob of classifing fake image to real image, we hope it small
        d_loss_fake = torch.mean(out_src)
        losses_d_fake.update(d_loss_fake.item(), input.size(0))

        # Compute loss for gradient penalty.
        alpha = torch.rand(x_real.size(0), 1, 1, 1).cuda()
        x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        out_src, _ = model_netD(x_hat)
        d_loss_gp = gradient_penalty(out_src, x_hat)

        # Backward and optimize.
        d_loss = d_loss_real + d_loss_fake + 10.0 * d_loss_gp + 1.0 * d_loss_cls
        losses_d.update(d_loss.item(), input.size(0))

        # optimizerG.zero_grad()
        optimizerD.zero_grad()
        d_loss.backward()
        optimizerD.step()

        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #
        if (i + 1) % 5 == 0:
            # Original-to-target domain.
            x_fake_delta = model_netG(x_real, label_trg)
            x_fake = x_real + 2.0 * x_fake_delta
            x_fake = torch.clamp(x_fake, -1.0, 1.0)

            out_src, out_cls = model_netD(x_fake)  # prob of classifying fake image to real image, we hope it large
            g_loss_fake = - torch.mean(out_src)
            losses_g_fake.update(g_loss_fake.item(), input.size(0))
            g_loss_cls = criterion_CEL(out_cls, label_trg)
            losses_g_cls.update(g_loss_cls.item(), input.size(0))

            # Target-to-original domain.
            x_reconst_delta = model_netG(x_fake, label_org)
            x_reconst = x_fake + 2.0 * x_reconst_delta
            x_reconst = torch.clamp(x_reconst, -1.0, 1.0)

            g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))
            losses_g_reconst.update(g_loss_rec.item(), input.size(0))

            # Constraint only a part region of the image is changed
            C = 0.1
            g_loss_sparsity = torch.mean(torch.abs(x_fake_delta)) + torch.mean(torch.abs(x_reconst_delta))
            g_loss_sparsity = torch.max(g_loss_sparsity - C, torch.zeros(1).cuda())

            # Compute loss for fooling the classifier to be explained
            predicted_labels = model_clser(x_fake)
            loss_cls = criterion_CEL(predicted_labels, label_trg)
            prec1, _ = accuracy(predicted_labels, label_trg, topk=(1, 5))
            top1.update(prec1[0], input.size(0))


            g_loss = g_loss_fake + loss_cls + g_loss_rec + g_loss_sparsity + g_loss_cls
            losses_g.update(g_loss.item(), input.size(0))

            optimizerG.zero_grad()
            # optimizerD.zero_grad()
            g_loss.backward()
            optimizerG.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            curr_lr = optimizerG.param_groups[0]['lr']
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: [{4}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_g {loss_g.val:.4f} ({loss_g.avg:.4f})\t'
                  'Loss_g_fake {loss_g_fake.val:.4f} ({loss_g_fake.avg:.4f})\t'
                  'Loss_g_rec {loss_g_rec.val:.4f} ({loss_g_rec.avg:.4f})\t'
                  'Loss_g_cls {loss_g_cls.val:.4f} ({loss_g_cls.avg:.4f})\t'
                  'Loss_d {loss_d.val:.4f} ({loss_d.avg:.4f})\t'
                  'Loss_d_real {loss_d_real.val:.4f} ({loss_d_real.avg:.4f})\t'
                  'Loss_d_fake {loss_d_fake.val:.4f} ({loss_d_fake.avg:.4f})\t'
                  'Loss_d_cls {loss_d_cls.val:.4f} ({loss_d_cls.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, args.epochs, i, len(train_loader), curr_lr,
                batch_time=batch_time, data_time=data_time, loss_g=losses_g, loss_g_fake=losses_g_fake, loss_g_rec=losses_g_reconst, loss_g_cls=losses_g_cls, loss_d=losses_d, loss_d_real=losses_d_real, loss_d_fake=losses_d_fake, loss_d_cls=losses_d_cls, top1=top1))



    # generate images for each possible

    if epoch > 98:

        if not os.path.exists("./capital_letter/demo" + "/source_image"):
            os.makedirs("./capital_letter/demo" + "/source_image")
        if not os.path.exists("./capital_letter/demo" + "/target_image"):
            os.makedirs("./capital_letter/demo" + "/target_image")
        if not os.path.exists("./capital_letter/demo" + "/delta_image"):
            os.makedirs("./capital_letter/demo" + "/delta_image")

        for i, (input, target, index) in enumerate(train_loader):

            if i > 1:
                break

            # measure data loading time
            data_time.update(time.time() - end)

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            x_real = input.cuda()
            label_org = target.cuda(async=True)
            label_trg_tmp = np.copy(target)
            for i_iter in range(26-1):
                label_trg_tmp = label_trg_tmp - 1
                label_trg_tmp[label_trg_tmp < 0] = 25

                label_trg = torch.from_numpy(label_trg_tmp).cuda()

                c_org = label2onehot(label_org, 26)
                c_trg = label2onehot(label_trg, 26)

                x_fake_delta = model_netG(x_real, label_trg)
                x_fake = x_real + 2.0 * x_fake_delta
                x_fake = torch.clamp(x_fake, -1.0, 1.0)

                x_real_tmp = (x_real + 1) / 2.0
                for i_img in range(c_trg.size(0)):

                    class_org = label_org[i_img]
                    class_org = class_org.data.cpu().numpy()
                    class_trg = label_trg[i_img]
                    class_trg = class_trg.data.cpu().numpy()

                    save_path = "./capital_letter/demo" + "/source_image/" + str(class_org) + "_" + str(class_trg) + "_" + str(index[i_img].data.cpu().numpy()).zfill(5) + ".png"
                    vutils.save_image(x_real_tmp[i_img].data, save_path, nrow=1)

                    save_path = "./capital_letter/demo" + "/target_image/" + str(class_org) + "_" + str(class_trg) + "_" + str(index[i_img].data.cpu().numpy()).zfill(5) + ".png"
                    vutils.save_image(x_fake[i_img].data, save_path, nrow=1)

                    save_path = "./capital_letter/demo" + "/delta_image/" + str(class_org) + "_" + str(class_trg) + "_" + str(index[i_img].data.cpu().numpy()).zfill(5) + ".png"
                    fake_delta_image = x_fake_delta[i_img].data.cpu().numpy()
                    fake_delta_image = fake_delta_image.squeeze()

                    final_fake_delta_image = np.zeros((28, 28, 3))
                    # final_fake_delta_image[:, :, 0] = x_real[i_img].data.cpu().numpy().squeeze()
                    # final_fake_delta_image[:, :, 1] = x_real[i_img].data.cpu().numpy().squeeze()
                    # final_fake_delta_image[:, :, 2] = x_real[i_img].data.cpu().numpy().squeeze()
                    final_fake_delta_image_B = x_real_tmp[i_img].data.cpu().numpy().squeeze()
                    final_fake_delta_image_B[fake_delta_image > 0.7] = fake_delta_image[fake_delta_image > 0.7]
                    final_fake_delta_image_B[fake_delta_image < -0.7] = 0
                    final_fake_delta_image_R = x_real_tmp[i_img].data.cpu().numpy().squeeze()
                    final_fake_delta_image_R[fake_delta_image < -0.7] = -fake_delta_image[fake_delta_image < -0.7]
                    final_fake_delta_image_R[fake_delta_image > 0.7] = 0
                    final_fake_delta_image_G = x_real_tmp[i_img].data.cpu().numpy().squeeze()
                    final_fake_delta_image_G[fake_delta_image > 0.7] = 0
                    final_fake_delta_image_G[fake_delta_image < -0.7] = 0
                    final_fake_delta_image[:, :, 0] = final_fake_delta_image_R
                    final_fake_delta_image[:, :, 2] = final_fake_delta_image_B
                    final_fake_delta_image[:, :, 1] = final_fake_delta_image_G

                    final_fake_delta_image = np.uint8(255 * final_fake_delta_image)
                    cv2.imwrite(save_path, final_fake_delta_image)


                    # fake_delta_image = -fake_delta_image
                    #
                    # plt.imshow(fake_delta_image, cmap='RdBu')
                    # plt.axis('off')
                    # plt.tight_layout()
                    # plt.savefig(save_path, bbox_inches='tight', transparent=True, pad_inches=0.0)
                    # plt.close()

                    # plt.imsave(save_path, fake_delta_image, cmap='RdBu', format="png")





def save_checkpoint(state, filename='checkpoint_res.pth.tar'):
    torch.save(state, filename)


def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).cuda()
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias,
    bn weights, bn bias, linear bias)
    Args:
        net: network architecture

    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)

        else:
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)

    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
