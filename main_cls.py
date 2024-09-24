from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from datasets.ModelNetDataset import ModelNet40, PointcloudScaleAndTranslate
from datasets.ModelNetDatasetFewShot import ModelNetFewShot
from datasets.ScanObjectNNDataset import ScanObjectNN_objectonly, ScanObjectNN_objectbg, ScanObjectNN_hardest
from models.model_cls import POS_BERT
import numpy as np
from torch.utils.data import DataLoader
from utils_downstream.util import cal_loss, IOStream
import sklearn.metrics as metrics
from torchvision import transforms
from models.transformer_util import index_points, furthest_point_sample

test_transforms = transforms.Compose(
    [
        PointcloudScaleAndTranslate(),
    ]
)

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = furthest_point_sample(data, number)
    fps_data = index_points(data, fps_idx.long())
    return fps_data


def _init_():
    # fix random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    # prepare file structures
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if not os.path.exists('./checkpoints/'+args.exp_name):
        os.makedirs('./checkpoints/'+args.exp_name)
    if not os.path.exists('./checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('./checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main_cls.py ./checkpoints/'+args.exp_name+'/main_cls.py.backup')
    os.system('cp models/model_cls.py ./checkpoints/'+args.exp_name+'/model_cls.py.backup')

def train(args, io):
    if args.dataset == 'modelnet40':
        train_dataset = ModelNet40(partition='train', num_points=args.num_points)
        test_dataset = ModelNet40(partition='test', num_points=args.num_points)
    elif args.dataset == 'objbg':
        train_dataset = ScanObjectNN_objectbg(partition='train')
        test_dataset = ScanObjectNN_objectbg(partition='test')
    elif args.dataset == 'objonly':
        train_dataset = ScanObjectNN_objectonly(partition='train')
        test_dataset = ScanObjectNN_objectonly(partition='test')
    elif args.dataset == 'objhard':
        train_dataset = ScanObjectNN_hardest(partition='train')
        test_dataset = ScanObjectNN_hardest(partition='test')
    elif args.dataset == 'modelnet40fewshot':
        train_dataset = ModelNetFewShot(partition='train', num_points=args.num_points, way=args.way, shot=args.shot, fold=args.fold)
        test_dataset = ModelNetFewShot(partition='test', num_points=args.num_points, way=args.way, shot=args.shot, fold=args.fold)
    train_loader = DataLoader(train_dataset, num_workers=6, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=6, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    io.cprint("Let's use" + str(torch.cuda.device_count()) + "GPUs!")
    
    # create model
    if 'modelnet40' in args.dataset:
        model = POS_BERT()
    elif 'obj' in args.dataset:
        model = POS_BERT(num_classes=15)
    else:
        raise NotImplementedError
    if args.model_path:
        state_dict = torch.load(args.model_path, map_location="cpu")
        state_dict = state_dict['teacher']
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.model_path, msg))
    if args.model_path_1:
        state_dict = torch.load(args.model_path_1, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.model_path, msg))
    model = model.to(device)
    model = nn.DataParallel(model)

    if args.optim == 'sgd':
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim == 'adam':
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adamw':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    # print(name)
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
        param_groups = add_weight_decay(model, weight_decay=args.wd)
        opt = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.wd)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-6 if args.optim == 'adam' else 1e-4)
    elif args.scheduler == 'step':
        scheduler = MultiStepLR(opt, [120, 160], gamma=0.1)
    elif args.scheduler == 'coslr':
        from timm.scheduler import CosineLRScheduler
        scheduler = CosineLRScheduler(opt,
                t_initial=args.epochs,
                t_mul=1,
                lr_min=1e-6,
                decay_rate=0.1,
                warmup_lr_init=1e-6,
                warmup_t=10,
                cycle_limit=1,
                t_in_epochs=True)
    
    criterion = cal_loss

    best_test_acc = 0
    best_votetest_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        if args.num_points == 1024:
            point_all = 1200
        elif args.num_points == 2048:
            point_all = 2400
        elif args.num_points == 4096:
            point_all = 4800
        elif args.num_points == 8192:
            point_all = 8192
        else:
            raise NotImplementedError()

        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            if data.size(1) < point_all:
                point_all = data.size(1)
            fps_idx = furthest_point_sample(data, point_all)  # (B, npoint)
            fps_idx = fps_idx[:, np.random.choice(point_all, args.num_points, False)]
            data = index_points(data, fps_idx.long())
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, lr: %.6f, loss: %.6f, train acc: %.6f' % (epoch, opt.param_groups[0]['lr'], train_loss*1.0/count,
                                                                metrics.accuracy_score(
                                                                    train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = fps(data, args.num_points)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f' % (epoch, test_loss*1.0/count, test_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), './checkpoints/%s/models/model.t7' % args.exp_name)
        io.cprint('best: %.3f' % best_test_acc)

        ####################
        # Vote Test
        ####################
        if args.vote and test_acc>0.91:
            votetest_loss = 0.0
            count = 0.0
            model.eval()
            votetest_pred = []
            votetest_true = []
            for data, label in test_loader:
                data, label = data.to(device), label.to(device).squeeze()
                if data.size(1) < point_all:
                    point_all = data.size(1)
                batch_size = data.size()[0]
                local_pred = []
                fps_idx_raw = furthest_point_sample(data, point_all)  # (B, npoint)
                for v in range(10):
                    fps_idx = fps_idx_raw[:, np.random.choice(point_all, args.num_points, False)]
                    new_data = index_points(data, fps_idx.long()) # (B, N, 3)
                    new_data = test_transforms(new_data)
                    new_data = new_data.permute(0, 2, 1)
                    logits = model(new_data)
                    local_pred.append(logits.detach().unsqueeze(0))
                pred = torch.cat(local_pred, dim=0).mean(0)
                loss = criterion(pred, label)
                _, preds = torch.max(pred, -1)
                count += batch_size
                votetest_loss += loss.item() * batch_size
                votetest_true.append(label.cpu().numpy())
                votetest_pred.append(preds.detach().cpu().numpy())
            votetest_true = np.concatenate(votetest_true)
            votetest_pred = np.concatenate(votetest_pred)
            votetest_acc = metrics.accuracy_score(votetest_true, votetest_pred)
            outstr = 'Vote Test %d, loss: %.6f, test acc: %.6f' % (epoch, votetest_loss*1.0/count, votetest_acc)
            io.cprint(outstr)
            if votetest_acc >= best_votetest_acc:
                best_votetest_acc = votetest_acc
                torch.save(model.state_dict(), './checkpoints/%s/models/votemodel.t7' % args.exp_name)
            io.cprint('vote_best: %.3f' % best_votetest_acc)

def test_vote(args, io):
    if args.dataset == 'modelnet40':
        test_dataset = ModelNet40(partition='test', num_points=args.num_points)
    elif args.dataset == 'objbg':
        test_dataset = ScanObjectNN_objectbg(partition='test')
    elif args.dataset == 'objonly':
        test_dataset = ScanObjectNN_objectonly(partition='test')
    elif args.dataset == 'objhard':
        test_dataset = ScanObjectNN_hardest(partition='test')
    elif args.dataset == 'modelnet40fewshot':
        test_dataset = ModelNetFewShot(partition='test', num_points=args.num_points, way=args.way, shot=args.shot, fold=args.fold)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if 'modelnet40' in args.dataset:
        model = POS_BERT().to(device)
    elif 'obj' in args.dataset:
        model = POS_BERT(num_classes=15).to(device)
    else:
        raise NotImplementedError
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))

    model = model.eval()

    best_vote_acc = 0.0

    for epoch in range(args.vote_epoch):
        randomseed = test_init(6156)
        test_true = []
        test_pred = []
        if args.num_points == 1024:
            point_all = 1200
            fps_num_points = 1024
        elif args.num_points == 2048:
            point_all = 2400
            fps_num_points = 2048
        elif args.num_points == 4096:
            point_all = 4800
            fps_num_points = 4096
        elif args.num_points == 8192:
            point_all = 8192
            fps_num_points = 7692
        else:
            raise NotImplementedError()
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            if data.size(1) < point_all:
                point_all = data.size(1)
            local_pred = []
            fps_idx_raw = furthest_point_sample(data, point_all)  # (B, npoint)
            for v in range(10):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, fps_num_points, False)]
                new_data = index_points(data, fps_idx.long())  # (B, N, 3)
                new_data = test_transforms(new_data)
                new_data = new_data.permute(0, 2, 1)
                logits = model(new_data)
                local_pred.append(logits.detach().unsqueeze(0))
            pred = torch.cat(local_pred, dim=0).mean(0)
            _, preds = torch.max(pred, -1)
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        if test_acc > best_vote_acc:
            best_vote_acc = test_acc
        if epoch>1:
            outstr = 'Test %d Seed %d :: test acc: %.6f, best test acc :%.6f'%(epoch, randomseed, test_acc, best_vote_acc)
            io.cprint(outstr)

def test(args, io):
    if args.dataset == 'modelnet40':
        test_dataset = ModelNet40(partition='test', num_points=args.num_points)
    elif args.dataset == 'objbg':
        test_dataset = ScanObjectNN_objectbg(partition='test')
    elif args.dataset == 'objonly':
        test_dataset = ScanObjectNN_objectonly(partition='test')
    elif args.dataset == 'objhard':
        test_dataset = ScanObjectNN_hardest(partition='test')
    elif args.dataset == 'modelnet40fewshot':
        test_dataset = ModelNetFewShot(partition='test', num_points=args.num_points, way=args.way, shot=args.shot, fold=args.fold)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if 'modelnet40' in args.dataset:
        model = POS_BERT().to(device)
    elif 'obj' in args.dataset:
        model = POS_BERT(num_classes=15).to(device)
    else:
        raise NotImplementedError
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))

    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    test_score = []
    if args.num_points == 1024:
        point_all = 1200
    elif args.num_points == 2048:
        point_all = 2400
    elif args.num_points == 4096:
        point_all = 4800
    elif args.num_points == 8192:
        point_all = 8192
    else:
        raise NotImplementedError()
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        if data.size(1) < point_all:
            point_all = data.size(1)
        data = fps(data, args.num_points)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        scores = logits.softmax(dim=-1)
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
        test_score.append(scores.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_score = np.concatenate(test_score)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    # test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    # test_roc_auc = metrics.roc_auc_score(test_true, test_score, average='macro', multi_class='ovo')
    outstr = 'Test :: test acc: %.6f'%(test_acc)
    io.cprint(outstr)


def multi_class_auc_roc(test_true, test_score, title='roc_curve.svg'):
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from scipy import interp

    y_test = label_binarize(test_true, classes=list(range(40)))
    y_score = test_score
    # 设置种类
    n_classes = y_test.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.3f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.3f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','tan','navy','pink','m','y','c',
                    'g','r','b','purple','brown','gray','lime'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.3f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.xlabel('False Positive Rate', font={'family': 'Times New Roman', 'size': '15'})
    plt.ylabel('True Positive Rate', font={'family': 'Times New Roman', 'size': '15'})
    plt.title('ROC curve of our method in ScanObjectNN classification task', font={'family': 'Times New Roman', 'size': '16'})
    plt.legend(loc="lower right", ncol=2, fontsize=8)
    plt.show()
    plt.savefig(title, dpi=1200, format='svg')




def test_init(seed=None):
    seed = np.random.randint(1, 10000) if seed==None else seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='POS_BERT', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40', 'objbg', 'objonly', 'objhard', 'modelnet40fewshot'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    # parser.add_argument('--use_sgd', type=bool, default=True,
    #                     help='Use SGD')
    parser.add_argument('--optim', type=str, default='sgd', metavar='N',
                        choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=0.0001, metavar='WD',
                        help='weight decay')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step', 'coslr'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--model_path_1', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--vote', type=bool, default=False,
                        help='enables Vote for test')
    parser.add_argument('--vote_epoch', type=int, default=1000,
                        help='num of points to use')
    parser.add_argument('--way', type=int, default=-1, choices=[-1, 5, 10])
    parser.add_argument('--shot', type=int, default=-1, choices=[-1, 10, 20])
    parser.add_argument('--fold', type=int, default=-1)
    args = parser.parse_args()

    seed = np.random.randint(1, 10000)

    _init_()

    if args.eval:
        io = IOStream('./checkpoints/' + args.exp_name + '/eval.log')
    else:
        io = IOStream('./checkpoints/' + args.exp_name + '/run.log')
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if not args.eval:
        train(args, io)
    else:
        with torch.no_grad():
            if args.vote:
                test_vote(args, io)
            else:
                test(args, io)
