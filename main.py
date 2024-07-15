import os
import shutil

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re
import time
import random
import logging
import datetime
import numpy as np

import model, train_and_test as tnt
import util.utils as utils
from util.utils import str2bool
from torch.utils.tensorboard import SummaryWriter
from util.engine_ppn import evaluate_joint
from pathlib import Path
from util.datasets import Cub2011AttributeWhole, Cub2011Eval
from util.preprocess import mean, std
from util.eval_concept_trustworthiness import get_activation_maps, evaluate_concept_trustworthiness


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_outlog(args):
    if args.eval: # evaluation only
        logfile_dir = os.path.join(args.output_dir, "eval-logs")
    else: # training
        logfile_dir = os.path.join(args.output_dir, "train-logs")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    tb_dir = os.path.join(args.output_dir, "tf-logs")
    tb_log_dir = os.path.join(tb_dir, args.base_architecture+ "_" + args.data_set)
    os.makedirs(logfile_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(
        log_dir=os.path.join(
            tb_dir,
            args.base_architecture+ "_" + args.data_set
        ),
        flush_secs=1
    )
    logger = utils.get_logger(
        level=logging.INFO,
        mode="w",
        name=None,
        logger_fp=os.path.join(
            logfile_dir,
            args.base_architecture+ "_" + args.data_set + ".log"
        )
    )

    logger = logging.getLogger("main")
    return tb_writer, logger


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1028)
parser.add_argument('--output_dir', default='output_debug/test/')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--resume', default='', help='resume from checkpoint')
# Data
parser.add_argument('--use_crop', type=str2bool, default=True)
parser.add_argument('--data_set', default='CUB2011',
    choices=['CUB2011A', 'CUB2011U', 'Car', 'Dogs', 'CUB2011', 'CUB2011AW', 'CUB2011AO'], type=str)
parser.add_argument('--data_path', type=str, default='datasets/cub200_cropped/')
parser.add_argument('--train_batch_size', default=80, type=int)
parser.add_argument('--test_batch_size', default=150, type=int)

# Model
parser.add_argument('--base_architecture', type=str, default='vgg16')
parser.add_argument('--input_size', default=224, type=int, help='images input size')
parser.add_argument('--save_ep_freq', default=400, type=int, help='save epoch frequency')
parser.add_argument('--prototype_shape', nargs='+', type=int, default=[2000, 64, 1, 1])
parser.add_argument('--prototype_activation_function', type=str, default='log')
parser.add_argument('--add_on_layers_type', type=str, default='regular')

# Loss
parser.add_argument('--use_mse_loss', type=str2bool, default=True)
parser.add_argument('--use_ortho_loss', type=str2bool, default=True)
parser.add_argument('--attri_coe', type=float, default=0.50)
parser.add_argument('--ortho_coe', type=float, default=1e-4)
parser.add_argument('--mse_coe', type=float, default=0.30)
parser.add_argument('--consis_coe', type=float, default=0.30)
parser.add_argument('--consis_thresh', type=float, default=0.10)
parser.add_argument('--cls_dis_coe', type=float, default=0.50)
parser.add_argument('--sep_dis_coe', type=float, default=0.50)

# Optimizer & Scheduler
parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER')
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
parser.add_argument('--features_lr', type=float, default=1e-4)
parser.add_argument('--add_on_layers_lr', type=float, default=3e-3)
parser.add_argument('--prototype_vectors_lr', type=float, default=3e-3)
parser.add_argument('--predictor_lr', type=float, default=1e-4)
parser.add_argument('--add_on_layers_final_lr', type=float, default=5e-4)
parser.add_argument('--prototype_vectors_final_lr', type=float, default=5e-4)
parser.add_argument('--epochs', type=int, default=22)
parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N')
parser.add_argument('--proto_epochs', type=int, default=12, metavar='N')
parser.add_argument('--decay_epochs', type=int, default=3)
parser.add_argument('--decay_rate', type=float, default=0.2)

# Distributed Training
parser.add_argument('--device', default='cuda', help='device to use for training / testing')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')

args = parser.parse_args()

__global_values__ = dict(it=0)
seed = args.seed + utils.get_rank()
set_seed(seed)

# Distributed Training
utils.init_distributed_mode(args)

tb_writer, logger = get_outlog(args)

if utils.get_rank() == 0:
    logger.info("Start running with args: \n{}".format(args))
device = torch.device(args.device)

# Setting Parameters
base_architecture = args.base_architecture
dataset_name = args.data_set

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
model_dir = args.output_dir

os.makedirs(model_dir, exist_ok=True)

shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'models', base_architecture_type + '_features_all.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

# Load the hyper param
if dataset_name == 'CUB2011' or dataset_name == 'CUB2011U':
    args.nb_classes = 200
elif dataset_name == 'Car':
    args.nb_classes = 196
img_size = args.input_size

# Optimzer
joint_optimizer_lrs = {'features': args.features_lr,
                    'add_on_layers': args.add_on_layers_lr,
                    'prototype_vectors': args.prototype_vectors_lr,
                    'predictor': args.predictor_lr}
warm_optimizer_lrs = {'add_on_layers': args.add_on_layers_lr,
                    'prototype_vectors': args.prototype_vectors_lr,
                    'predictor': args.predictor_lr}

coefs = {
    'crs_ent': 1,
    'attri': args.attri_coe,
    'orth': 1e-4,
    'mse': args.mse_coe,
    'consis': args.consis_coe,
    'cls_dis': args.cls_dis_coe,
    'sep_dis': args.sep_dis_coe,
}

normalize = transforms.Normalize(mean=mean, std=std)

transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ])
train_dataset = Cub2011AttributeWhole(train=True, transform=transform)
test_dataset = Cub2011AttributeWhole(train=False, transform=transform)
test_loc_dataset = Cub2011Eval(root='datasets/cub200_cropped/', train=False, transform=transform)
args.nb_classes = train_dataset.nb_classes

if args.distributed:
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    if args.dist_eval:
        if len(test_dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(test_dataset)
        sampler_val_loc = torch.utils.data.SequentialSampler(test_loc_dataset)
else:
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    sampler_val_loc = torch.utils.data.SequentialSampler(test_loc_dataset)

# Train loader & test loader
train_loader = torch.utils.data.DataLoader(
    train_dataset, sampler=sampler_train,
    batch_size=args.train_batch_size,
    num_workers=16, 
    pin_memory=False)
test_loader = torch.utils.data.DataLoader(
    test_dataset, sampler=sampler_val,
    batch_size=args.test_batch_size,
    num_workers=16,
    pin_memory=False)
test_loc_loader = torch.utils.data.DataLoader(
    test_loc_dataset, sampler=sampler_val_loc,
    batch_size=args.test_batch_size,
    num_workers=16,
    pin_memory=False)

# Construct the model
ppnet = model.construct_CBMNet(base_architecture=args.base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=args.prototype_shape,
                              num_classes=args.nb_classes,
                              prototype_activation_function=args.prototype_activation_function,
                              add_on_layers_type=args.add_on_layers_type)
ppnet.to(device)
ppnet_without_ddp = ppnet
if args.distributed:
    ppnet = torch.nn.parallel.DistributedDataParallel(ppnet, device_ids=[args.gpu], find_unused_parameters=True)
    ppnet_without_ddp = ppnet.module
n_parameters = sum(p.numel() for p in ppnet.parameters() if p.requires_grad)
if utils.get_rank() == 0:
    logger.info('number of params: {}'.format(n_parameters))

if args.resume:
    checkpoint = torch.load(args.resume, map_location='cpu')
    # ppnet_without_ddp.load_state_dict(checkpoint['model'])
    ppnet_without_ddp.load_state_dict(checkpoint['model'], strict=False)

# Define optimizer
joint_optimizer_specs = \
[{'params': ppnet_without_ddp.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet_without_ddp.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet_without_ddp.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=args.decay_epochs, gamma=args.decay_rate)

warm_optimizer_specs = \
[{'params': ppnet_without_ddp.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet_without_ddp.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

final_optimizer_specs = \
[
 {'params': ppnet_without_ddp.add_on_layers.parameters(), 'lr': args.add_on_layers_final_lr, 'weight_decay': 1e-3},
 {'params': ppnet_without_ddp.prototype_vectors, 'lr': args.prototype_vectors_final_lr},
 {'params': ppnet_without_ddp.attributes_predictor.parameters(), 'lr': joint_optimizer_lrs['predictor']},
 {'params': ppnet_without_ddp.class_predictor.parameters(), 'lr': joint_optimizer_lrs['predictor']},
]
final_optimizer = torch.optim.Adam(final_optimizer_specs)
final_lr_scheduler = torch.optim.lr_scheduler.StepLR(final_optimizer, step_size=args.decay_epochs, gamma=args.decay_rate)

output_dir = Path(args.output_dir)

# Train the model
if utils.get_rank() == 0:
    logger.info(f"Start training for {args.epochs} epochs")
start_time = time.time()
for epoch in range(args.epochs):
    if epoch < args.warmup_epochs:
        tnt.warm_only_new(model=ppnet)
        _, train_results = tnt.train(model=ppnet, epoch=epoch, dataloader=train_loader, optimizer=warm_optimizer,
                      coefs=coefs, args=args, tb_writer=tb_writer, iteration=__global_values__["it"])
        continue
    elif epoch < args.proto_epochs:
        tnt.joint_new(model=ppnet)
        joint_lr_scheduler.step()
        _, train_results = tnt.train(model=ppnet, epoch=epoch, dataloader=train_loader, optimizer=joint_optimizer,
                      coefs=coefs, args=args, tb_writer=tb_writer, iteration=__global_values__["it"])
        continue
    else:
        tnt.final_new(model=ppnet)
        final_lr_scheduler.step()
        _, train_results = tnt.train(model=ppnet, epoch=epoch, dataloader=train_loader, optimizer=final_optimizer,
                      coefs=coefs, args=args, tb_writer=tb_writer, iteration=__global_values__["it"])

    test_stats = evaluate_joint(data_loader=test_loader, model=ppnet, device=device, args=args, epoch=epoch)
    
    if epoch >= args.proto_epochs:
        all_activation_maps, all_img_ids = get_activation_maps(ppnet, test_loc_loader)
        mean_loc_acc, (all_loc_acc, all_attri_idx, all_num_samples) = evaluate_concept_trustworthiness(all_activation_maps, all_img_ids, bbox_half_size=45)
    else:
        mean_loc_acc = 0.0
    tb_writer.add_scalar("epoch/val_acc1", test_stats['acc1'], epoch)
    tb_writer.add_scalar("epoch/val_loss", test_stats['loss'], epoch)
    tb_writer.add_scalar("epoch/val_acc5", test_stats['acc5'], epoch)

    if utils.get_rank() == 0:
        logger.info(f"Accuracy of the network on the {len(test_dataset)} test images: {test_stats['acc1']:.2f}%")
        logger.info(f"Concept trustworthiness score of the network on the {len(test_dataset)} test images: {mean_loc_acc:.2f}%")
        logger.info(f"\n")
    if epoch == args.epochs - 1:   # save the last
        checkpoint_paths = [output_dir / 'checkpoints/epoch-last.pth']
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': ppnet_without_ddp.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
    if utils.get_rank() == 0:
        logger.info(f'')

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
if utils.get_rank() == 0:
    logger.info('Training time {}'.format(total_time_str))