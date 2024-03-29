"""
author : takiKim
"""

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os, re
from glob import glob
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as torch_multiprocessing


class ImageDataset(Dataset):
    def __init__(self, img_size, dataset_path):
        self.train_images = self.listdir(dataset_path)
        self.train_labels = []  # make label

        # interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        transform_list = [
            transforms.Resize(size=[img_size, img_size]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),  # [0, 255] -> [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),  # [0, 1] -> [-1, 1]
        ]

        self.transform = transforms.Compose(transform_list)

    def listdir(self, dir_path):
        extensions = ['png', 'jpg', 'jpeg', 'JPG']
        file_path = []
        for ext in extensions:
            file_path += glob(os.path.join(dir_path, '*.' + ext))
        file_path.sort()
        return file_path

    def __getitem__(self, index):
        sample_path = self.train_images[index]
        img = Image.open(sample_path).convert('RGB')
        img = self.transform(img)

        label = self.train_labels[index]

        return img, label

    def __len__(self):
        return len(self.train_images)


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


# def multi_gpu_run(ddp_fn, args):  # in main
#     # ddp_fn = train_fn
#     world_size = torch.cuda.device_count()  # ngpus
#     torch_multiprocessing.spawn(fn=ddp_fn, args=(args, world_size), nprocs=world_size, join=True)


# def build_init_procss(rank, world_size, device):  # in build
#     os.environ["MASTER_ADDR"] = "127.0.0.1"  # localhost
#     os.environ["MASTER_PORT"] = "12355"
#
#     # initialize the process group
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)
#     synchronize()
#     torch.cuda.set_device(device)


# def synchronize():
#     if not dist.is_available():
#         return
#
#     if not dist.is_initialized():
#         return
#
#     world_size = dist.get_world_size()
#
#     if world_size == 1:
#         return
#
#     dist.barrier()
#
#
# def dataparallel_and_sync(model, local_rank, find_unused_parameters=False):
#     # DistributedDataParallel
#     model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
#                                     find_unused_parameters=find_unused_parameters)
#
#     # broadcast
#     broadcast_params(model)
#
#     model = model.module
#
#     return model
#
#
# def broadcast_params(model):
#     params = model.parameters()
#     for param in params:
#         dist.broadcast(param.data, src=0)
#     dist.barrier()
#     torch.cuda.synchronize()
#
#
# def get_world_size():
#     if not dist.is_available():
#         return 1
#
#     if not dist.is_initialized():
#         return 1
#
#     return dist.get_world_size()
#
#
# def distributed_sampler(dataset, rank, num_replicas, shuffle):
#     return torch.utils.data.distributed.DistributedSampler(dataset, rank=rank, num_replicas=num_replicas,
#                                                            shuffle=shuffle)
#
#
# def infinite_iterator(loader):
#     while True:
#         for batch in loader:
#             yield batch
#
#
# def find_latest_ckpt(folder):
#     files = []
#     for fname in os.listdir(folder):
#         s = re.findall(r'\d+', fname)
#         if len(s) == 1:
#             files.append((int(s[0]), fname))
#     if files:
#         file_name = max(files)[1]
#         index = os.path.splitext(file_name)[0]
#         return file_name, index
#     else:
#         return None, 0
#
#
# def requires_grad(model, flag=True):
#     for p in model.parameters():
#         p.requires_grad = flag
#

def apply_gradients(loss, optim):
    optim.zero_grad()
    loss.backward()
    optim.step()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def reduce_loss_dict(loss_dict):
#     world_size = get_world_size()
#
#     if world_size < 2:
#         return loss_dict
#
#     with torch.no_grad():
#         keys = []
#         losses = []
#
#         for k in sorted(loss_dict.keys()):
#             keys.append(k)
#             losses.append(loss_dict[k])
#
#         losses = torch.stack(losses, 0)
#         dist.reduce(losses, dst=0)
#
#         if dist.get_rank() == 0:
#             losses /= world_size
#
#         reduced_losses = {k: v.mean().item() for k, v in zip(keys, losses)}
#
#     return reduced_losses


def cross_entroy_loss(logit, label):
    loss = torch.nn.CrossEntropyLoss()(logit, label)
    return loss


def check_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    return device
