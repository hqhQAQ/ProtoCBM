# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import logging
import pickle
from typing import Iterable, Optional

import torch
import numpy as np
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from torch.nn.modules.loss import _Loss
import util.utils as utils
from torch.utils.tensorboard import SummaryWriter


@torch.no_grad()
def get_img_mask(data_loader, model, device, args):
    logger = logging.getLogger("get mask")
    logger.info("Get mask")
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Get Mask:'

    # switch to evaluation mode
    model.eval()

    all_attn_mask = []
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            cat_mask = model.get_attn_mask(images)
            all_attn_mask.append(cat_mask.cpu())
    all_attn_mask = torch.cat(all_attn_mask, dim=0) # (num, 2, 14, 14)
    if hasattr(model, 'module'):
        model.module.all_attn_mask = all_attn_mask
    else:
        model.all_attn_mask = all_attn_mask


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    logger = logging.getLogger("validate")
    logger.info("Start validation")
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for data_item in metric_logger.log_every(data_loader, 10, header):
        if len(data_item) == 2:
            images, target = data_item
        else:
            images, target, attributes = data_item

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        # with torch.cuda.amp.autocast():
        with torch.no_grad():
            output, _ = model(images)
            if isinstance(output, tuple):
                _, output, _ = output
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        _, pred = output.topk(k=1, dim=1)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_joint(data_loader, model, device, args, epoch):
    logger = logging.getLogger("validate")
    logger.info("Start validation")
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for data_item in metric_logger.log_every(data_loader, 10, header):
        if len(data_item) == 2:
            images, target = data_item
        else:
            images, target, attributes = data_item

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        # with torch.cuda.amp.autocast():
        with torch.no_grad():
            output, _ = model(images)
            if isinstance(output, tuple):
                if epoch < args.proto_epochs:
                    output, _, _ = output
                else:
                    _, output, _ = output
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        _, pred = output.topk(k=1, dim=1)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_logits(data_loader, model, device, args):
    logger = logging.getLogger("validate")
    logger.info("Start validation")
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for data_item in metric_logger.log_every(data_loader, 10, header):
        if len(data_item) == 2:
            images, target = data_item
        else:
            images, target, attributes = data_item

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        # with torch.cuda.amp.autocast():
        with torch.no_grad():
            output, _ = model.forward_logits(images)
            if isinstance(output, tuple):
                _, output, _ = output
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        _, pred = output.topk(k=1, dim=1)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}