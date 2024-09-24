# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import models as torchvision_models
from tensorboardX import SummaryWriter

import utils
import point_transformer as pits
import pointcloudNets as pcns
from point_transformer import DINOHead

from dataloader import ModelNet40_STRL, DataAugmentationDINO, DataAugmentationSSTest, ShapeNet

from utils_3d.check_grad import validate_gradient
from utils_3d.eval_svm_utils import extract_features, evaluate_svm

from fvcore.nn import FlopCountAnalysis

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO-3D', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='pct_base', type=str,
                        choices=['pointnet', 'pointnet2', 'pointnet2_glr', 'dgcnn_large_k40',
                                 'dgcnn', 'dgcnn_large', 'pct_base', 'pct_large', 'pct_office',
                                 'pointnet_large', 'pointnet2_glr_large', 'pointnet2_large',
                                 'dgcnn_large_semseg', 'votenet_backbone'],
                        help="""Name of architecture to train""")
    parser.add_argument('--pointcloud_size', default=1024, type=int, help="""The number of Point cloud all point.""")
    parser.add_argument('--patch_size', default=32, type=int, help="""The number of Point Patch.""")
    parser.add_argument('--out_dim', default=512, type=int, help="""Dimensionality of the DINO head output.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
                        Not normalizing leads to better performance but can make the training unstable.
                        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
                        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
                        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--use_npair_loss', action='store_true', help='Use npair loss to train the model')

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
                        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
                        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
                        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
                        to use half precision for training. Improves training time and memory requirements,
                        but can provoke instability and slight decay of performance. We recommend disabling
                        mixed precision if the loss is unstable, if reducing the patch size or if training 
                        with bigger Models. (Default: off[False])""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
                        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
                        weight decay. We use a cosine schedule for WD and using a larger decay by
                        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
                        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
                        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=300, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
                        during which we keep the output layer fixed. Typically doing so during
                        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.005, type=float, help="""Learning rate at the end of
                        linear warmup (highest LR used during training). The learning rate is linearly scaled
                        with the batch size, and specified here for a reference batch size of 64.""")
    parser.add_argument("--warmup_epochs", default=20, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-5, help="""Target LR at the
                        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # loss
    parser.add_argument('--mpmloss', type=float, default=1.0, help="""loss1 weight.""")
    parser.add_argument('--gfcloss', type=float, default=0.5, help="""loss2 weight.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_ratio', type=float, nargs='+', default=(0.5, 0.9),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
                        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
                        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
                        local views to generate. Set this parameter to 0 to disable multi-crop training.
                        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_ratio', type=float, nargs='+', default=(0.2, 0.5),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
                        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--diff_resolution', type=int, nargs='+', default=(512, 256, 128, 64),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
                        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--diff_resolution_num', type=int, nargs='+', default=(2, 2, 2, 2),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
                        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--global2_weakly_aug', type=utils.bool_flag, default=False, help="""Whether or not
                        to use global2 weakly aug. (Default: off[False])""")

    # mask type
    parser.add_argument('--mask_type', default="rand", type=str, help='mask patch type.')
    parser.add_argument('--mask_ratio', type=float, nargs='+', default=(0.2, 0.4), help='mask ratio.')

    # Misc
    # parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
    #                     help='Please specify path to the Point cloud training data.')
    parser.add_argument('--output_dir', default="outputs_3D", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--exp_name', default="debug_exp", type=str, help='the name of this exp.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--print_freq', type=int, default=10, help="""Print txt log and write tensorboard log every x iter""")
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
                        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ tensorboard_writer init ============
    tensorboard_writer = SummaryWriter(log_dir=args.output_dir)

    # ============ output code version info and init the log ============
    if utils.is_main_process():
        with (Path(args.output_dir) / "log.txt").open("a") as f:
            f.write("git:\n  {}\n".format(utils.get_sha()))
            f.write("\n".join("%s: %s" % (k, str(v))
                              for k, v in sorted(dict(vars(args)).items())))

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(args.local_crops_number, args.diff_resolution, args.diff_resolution_num)
    dataset = ShapeNet(num_points=args.pointcloud_size,
                         partition='train',
                         transform=transform,
                         deterministic=False,
                         global_keep=args.global_crops_ratio,
                         local_keep=args.local_crops_ratio
                         )
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded: there are {len(dataset)} Point cloud.")

    # ============ preparing val data ... ============
    transform_val = DataAugmentationSSTest()
    dataset_train = ReturnIndexDataset(num_points=args.pointcloud_size,
                                       partition='train',
                                       transform=transform_val,
                                       deterministic=True,
                                       global_keep=(1.0, 1.0))
    dataset_val = ReturnIndexDataset(num_points=args.pointcloud_size,
                                     partition='test',
                                     transform=transform_val,
                                     deterministic=True,
                                     global_keep=(1.0, 1.0))
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    sampler_val = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Val Data loaded with {len(dataset_train)} train and {len(dataset_val)} val Point cloud.")

    # ============ building student and teacher networks ... ============
    # if the network is a Point Cloud Transformer (i.e. pct_base)
    if args.arch in pits.__dict__.keys():
        student = pits.__dict__[args.arch](samplepoint=args.patch_size, mask_type=args.mask_type, mask_ratio=args.mask_ratio,  masked_im_modeling=True)
        teacher = pits.__dict__[args.arch](samplepoint=args.patch_size)
        embed_dim = student.embed_dim
    elif args.arch in pcns.__dict__.keys():
        student = pcns.__dict__[args.arch]()
        teacher = pcns.__dict__[args.arch]()
        embed_dim = student.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper3D(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper3D(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    # tensor = (torch.rand(1, 3, 2048).cuda(),)
    # # 分析FLOPs
    # flops = FlopCountAnalysis(student, tensor)
    # print("FLOPs: ", flops.total())

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], broadcast_buffers=False)
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], broadcast_buffers=False)
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2 + sum(args.diff_resolution_num),  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        use_npair_loss=args.use_npair_loss,
        mpmloss=args.mpmloss,
        gfcloss=args.gfcloss,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 64.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0, "best_loss": 1e5, "best_acc": 0.0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint_bestacc.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]
    best_loss = to_restore["best_loss"]
    best_acc = to_restore["best_acc"]

    start_time = time.time()
    print("Starting DINO_3D training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        epoch_start_time = time.time()
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, tensorboard_writer, args)
        epoch_end_time = time.time()



        # ============ valing one epoch of DINO ... ============
        # ============ extract features ... ============
        print("Extracting features for train set...")
        train_features = extract_features(teacher, data_loader_train)
        print("Extracting features for val set...")
        test_features = extract_features(teacher, data_loader_val)
        train_labels = torch.tensor([s[-1] for s in dataset_train.label]).long()
        test_labels = torch.tensor([s[-1] for s in dataset_val.label]).long()

        if utils.is_main_process():
            print("Features are ready!\nStart the SVM classification.")
            train_features = nn.functional.normalize(train_features, dim=1, p=2)
            test_features = nn.functional.normalize(test_features, dim=1, p=2)
            svm_acc= evaluate_svm(train_features.cpu().numpy(), train_labels.cpu().numpy(),
                                  test_features.cpu().numpy(), test_labels.cpu().numpy())

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'best_loss': train_stats['loss'],
            'dino_loss': dino_loss.state_dict(),
            'best_acc': svm_acc if utils.is_main_process() else None,
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if best_loss>train_stats['loss']:
            best_loss = train_stats['loss']
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint_bestloss.pth'))
        if utils.is_main_process():
            if best_acc < svm_acc:
                best_acc = svm_acc
                utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint_bestacc.pth'))
        # if args.saveckp_freq and epoch % args.saveckp_freq == 0:
        #     utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        time_epoch = epoch_end_time - epoch_start_time
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'cur_acc': svm_acc if utils.is_main_process() else None, 'best_acc': best_acc,
                     'epoch': epoch, 'epoch_time': time_epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, tensorboard_writer, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (pointclouds, _) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        pointclouds = [pc.cuda(non_blocking=True) for pc in pointclouds]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # iter_start_time = time.time()
            teacher_output = teacher(pointclouds[1:3])  # only the 2 global views pass through the teacher
            student_output_global = student(pointclouds[1:3])
            student.masked_im_modeling = False
            student_output_local = student(pointclouds[3:], False)
            # iter_end_time = time.time()
            # print("forward time: ", iter_end_time - iter_start_time)
            loss = dino_loss(student_output_global['global_feat'], teacher_output['global_feat'],
                             student_output_global['patch_feat'], teacher_output['patch_feat'],
                             student_output_local['global_feat'], student_output_global['bool_mask'], epoch)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            loss_fp16 = fp16_scaler.scale(loss)
            loss_fp16.backward()

            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            names_q, params_q, names_k, params_k = [], [], [], []
            for name_q, param_q in student.module.named_parameters():
                names_q.append(name_q)
                params_q.append(param_q)
            for name_k, param_k in teacher_without_ddp.named_parameters():
                names_k.append(name_k)
                params_k.append(param_k)
            names_common = list(set(names_q) & set(names_k))
            params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
            params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        if utils.is_main_process() and it%args.print_freq==0:
            for k, v in metric_logger.meters.items():
                tensorboard_writer.add_scalar(f'train/{k}-avg', v.avg, it)
                tensorboard_writer.add_scalar(f'train/{k}-value', v.value, it)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, use_npair_loss=False, mpmloss=1.0, gfcloss=0.5):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.use_npair_loss = use_npair_loss
        # self.ce = nn.CrossEntropyLoss()
        self.mpmloss = mpmloss
        self.gfcloss = gfcloss

    def forward(self, student_output, teacher_output, student_patch, teacher_patch, student_output_local, student_patchmask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(2)
        student_out_local = student_output_local / self.student_temp
        student_out_local = student_out_local.chunk(self.ncrops-2)
        student_out=student_out+student_out_local
        student_patch_c = student_patch / self.student_temp
        student_patch_c = student_patch_c.chunk(2)
        student_patchmask_c = student_patchmask.chunk(2)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(2)

        total_loss1, total_loss2 = 0, 0
        n_loss_terms1, n_loss_terms2 = 0, 0
        for q in range(len(teacher_out)):
            for v in range(len(student_out)):
                if v == q:
                    loss = torch.sum(-teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1), dim=-1)
                    mask_s = student_patchmask_c[v]
                    loss = torch.sum(loss * mask_s.float(), dim=-1) / mask_s.sum(dim=-1).clamp(min=1.0)
                    total_loss2 = total_loss2 + loss.mean()
                    n_loss_terms2 += 1
                else:
                    loss = torch.sum(-teacher_out[q] * F.log_softmax(student_out[v], dim=-1), dim=-1)
                    total_loss1 += loss.mean()
                    n_loss_terms1 += 1
        total_loss1 /= n_loss_terms1
        total_loss2 /= n_loss_terms2

        # compute pointglr npair-loss
        if self.use_npair_loss:
            student_out_for_npair = torch.cat([out_i[:, :, None] for out_i in student_out], dim=2)
            npair_loss = 0
            npair_loss_terms = 0
            for iq, q in enumerate(teacher_out):
                loss_iq = self.get_npair_loss(q, student_out_for_npair)
                npair_loss_terms += 1
                npair_loss += loss_iq
            npair_loss /= npair_loss_terms
            self.update_center(teacher_output)
            return (total_loss1 + total_loss2 + 0.01*npair_loss) / 3

        self.update_center(teacher_output, teacher_patch)
        return self.gfcloss*total_loss1 + self.mpmloss*total_loss2

    @torch.no_grad()
    def update_center(self, teacher_output, teacher_patch):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        batch_patch_center = torch.sum(teacher_patch, dim=0, keepdim=True)
        dist.all_reduce(batch_patch_center)
        batch_patch_center = batch_patch_center / (len(teacher_patch) * dist.get_world_size())

        # ema update
        self.center2 = self.center2 * self.center_momentum + batch_patch_center * (1 - self.center_momentum)

    def get_npair_loss(self, x, ref):
        '''
        :param x: (bs, n_rkhs)
        :param ref: (bs, n_rkhs, n_loc)
        :return: loss
        '''

        bs, n_rkhs, n_loc = ref.size()
        ref = ref.transpose(0, 1).reshape(n_rkhs, -1)
        score = torch.matmul(x, ref) * 64.  # (bs * n_loc, bs)
        score = score.view(bs, bs, n_loc).transpose(1, 2).reshape(bs * n_loc, bs)
        gt_label = torch.arange(bs, dtype=torch.long, device=x.device).view(bs, 1).expand(bs, n_loc).reshape(-1)
        return F.cross_entropy(score, gt_label)


class ReturnIndexDataset(ModelNet40_STRL):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx

@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 1000):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        if multiscale:
            feats = utils.multi_scale(samples, model) #pc需要重写
        else:
            feats = model(samples)['backbone_feat'].clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    model.train()
    return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
