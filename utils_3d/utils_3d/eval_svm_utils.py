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
import os
import sys
import argparse, json
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from sklearn.svm import LinearSVC
import numpy as np

import utils
import point_transformer as pits
import pointcloudNets as pcns
from dataloader import ModelNet40, DataAugmentationDINO, DataAugmentationSSTest
from dataloader import SaveRawPoint, RandomJitter, RandomCrop_Global_spherical, \
    RandomTransformSE3_euler, CompletionGlobalPoints_interp, ShufflePoints, OutputPoints


def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = DataAugmentationSSTest()
    dataset_train = ReturnIndexDataset(num_points=args.pointcloud_size,
                                       partition='train',
                                       transform=transform,
                                       deterministic=True,
                                       global_keep=args.global_crops_ratio)
    dataset_val = ReturnIndexDataset(num_points=args.pointcloud_size,
                                     partition='test',
                                     transform=transform,
                                     deterministic=True,
                                     global_keep=args.global_crops_ratio)
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
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    if args.arch in pits.__dict__.keys():
        model = pits.__dict__[args.arch](samplepoint=args.patch_size)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif args.arch in pcns.__dict__.keys():
        model = pcns.__dict__[args.arch]()
        print(f"Model {args.arch} built.")
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val, args.use_cuda)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train.label]).long()
    test_labels = torch.tensor([s[-1] for s in dataset_val.label]).long()
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        Path(args.dump_features).mkdir(parents=True, exist_ok=True)
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        if multiscale:
            feats = utils.multi_scale(samples, model) #pc需要重写
        else:
            feats = model(samples).clone()

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
    return features


@torch.no_grad()
def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


class ReturnIndexDataset(ModelNet40):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on Modelnet40')
    # dataloader
    parser.add_argument('--pointcloud_size', default=1024, type=int, help="""The number of Point cloud all point.""")
    parser.add_argument('--patch_size', default=32, type=int, help='Patch point cloud of the model.')
    parser.add_argument('--global_crops_ratio', type=float, nargs='+', default=(1.0, 1.0),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
                        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
                        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')

    # output dir
    parser.add_argument('--output_dir', default="outputs_3D", help='Path to save logs and checkpoints')
    parser.add_argument('--exp_name', default="debug_exp", help='Path to save logs and checkpoints')
    parser.add_argument('--eval_name', default="eval_svm", help='Path to save logs and checkpoints')
    parser.add_argument('--dump_features', default=None,
                        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None,
                        help="""If the features have already been computed, where to find them.""")

    # Network
    parser.add_argument('--arch', default='pct_base', type=str, help='Architecture')
    parser.add_argument('--pretrained_weights', default='', type=str,
                        help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')

    # KNN config
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature used in the voting coefficient')
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
                        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")

    # distributed
    parser.add_argument("--dist_url", default="env://", type=str,
                        help="""url used to set up distributed training; 
                        see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    args = parser.parse_args()

    # 根据exp-name和eval-name重新设定output-dir, 并创建output-dir
    args.output_dir = os.path.join(args.output_dir, args.exp_name, args.eval_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # ============ output code version info and para ============
    if utils.is_main_process():
        with (Path(args.output_dir) / "log_eval_svm.txt").open("a") as f:
            f.write("git:\n  {}\n".format(utils.get_sha()))
            f.write("\n".join("%s: %s" % (k, str(v))
                              for k, v in sorted(dict(vars(args)).items())))

    if args.load_features:
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
    else:
        # need to extract features !
        train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args)

    if utils.get_rank() == 0:
        if args.use_cuda:
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()

        print("Features are ready!\nStart the SVM classification.")
        top1= evaluate_svm(train_features.cpu().numpy(), train_labels.cpu().numpy(),
                           test_features.cpu().numpy(), test_labels.cpu().numpy())
        print_info = f"SVM classifier result: Top1: {top1}"
        print(print_info)
        if utils.is_main_process():
            with (Path(args.output_dir) / "log_eval_svm.txt").open("a") as f:
                f.write('\n'+json.dumps(print_info))
    dist.barrier()
