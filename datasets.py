import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import os
from PIL import Image



_NUM_CLASSES = {
    'mnist': 10,
    'capital_letter': 26,
    'chinese1': 10,
    'tiny_letter': 10,
}




def capital_letter(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/letter_gt_tr.txt'
    val_list = './datasets/letter_gt_te.txt'
    # val_list = './datasets/letter_gt_tr.txt'

    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist_MNIST(
                flist=train_list,
                transform=transforms.Compose([
                    # transforms.Pad(4),
                    transforms.Resize((28, 28)),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist_MNIST(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def tiny_letter(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/tiny_letter_gt_tr.txt'
    val_list = './datasets/tiny_letter_gt_te.txt'
    # val_list = './datasets/letter_gt_tr.txt'

    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist_MNIST(
                flist=train_list,
                transform=transforms.Compose([
                    # transforms.Pad(4),
#                    transforms.Resize((28, 28)),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist_MNIST(
                flist=val_list,
                transform=transforms.Compose([
#                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds



def chinese1(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/Chinese_char2_gt_tr.txt'
    val_list = './datasets/Chinese_char2_gt_te.txt'

    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist_MNIST(
                flist=train_list,
                transform=transforms.Compose([
                    # transforms.Pad(4),
                    # transforms.RandomCrop(28),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist_MNIST(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def mnist(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/mnist_png/mnist_gt_tr.txt'
    val_list = './datasets/mnist_png/mnist_gt_te.txt'

    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist_MNIST(
                flist=train_list,
                transform=transforms.Compose([
                    # transforms.Pad(4),
                    # transforms.RandomCrop(28),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist_MNIST(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds




def default_loader(path):
    return Image.open(path).convert('RGB')


def default_loader_mnist(path):
    return Image.open(path).convert('L')

# def default_loader(path):
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel, imindex = line.strip().split()
            imlist.append((impath, int(imlabel), int(imindex)))

    return imlist


def string_concat(input):
    output = ''
    for i in range(len(input)):
        output = output + " " + input[i]
    return output


def default_flist_reader_chinese(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            iminfo = line.strip().split()
            imlabel = iminfo[-2]
            imindex = iminfo[-1]
            impath = string_concat(iminfo[:-2])
            imlist.append((impath, int(imlabel), int(imindex)))

    return imlist


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray



class ImageFilelist(data.Dataset):
    def __init__(self, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target, index = self.imlist[index]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imlist)


class ImageFilelist_MNIST(data.Dataset):
    def __init__(self, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader_mnist):
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target, index = self.imlist[index]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imlist)

class ImageFilelist_CHINESE(data.Dataset):
    def __init__(self, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader_chinese, loader=default_loader_mnist):
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target, index = self.imlist[index]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imlist)


