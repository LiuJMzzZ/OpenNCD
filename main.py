import argparse
import warnings
import torch
import torch.nn.functional as F
import torch.optim as optim
import dataset.open_world_cifar as datasets
from utils import cluster_acc, AverageMeter, accuracy, Logger, TransformTwice, setup_seed, proto_graph, graph_cluster, reknn_graph
from sklearn import metrics
import numpy as np
import os
import sys
from itertools import cycle
from models.model import Model
import time
from scipy.optimize import linear_sum_assignment



def train(args, model, device, train_label_loader, train_unlabel_loader, optimizer, epoch):
    model.train()
    unlabel_loader_iter = cycle(train_unlabel_loader)
    ent_losses = AverageMeter('ent_loss', ':.4e')
    cls_losses = AverageMeter('cls_loss', ':.4e')
    group_losses = AverageMeter('group_loss', ':.4e')
    proto_losses = AverageMeter('proto_loss', ':.4e')

    for batch_idx, ((x_l, x_l2), y_l) in enumerate(train_label_loader):
        
        ((x_u, x_u2), y_u) = next(unlabel_loader_iter)
        x_l, y_l, x_u, y_u = x_l.to(device), y_l.to(device), x_u.to(device), y_u.to(device)
        x_l2, x_u2 = x_l2.to(device), x_u2.to(device)

        loss_dic = model.loss(x_l, x_l2, y_l, x_u, x_u2)
        loss =  loss_dic['proto'] + loss_dic['group'] +  args.reg[0] * loss_dic['ent'] + args.reg[1] * loss_dic['cls']

        proto_losses.update(loss_dic['proto'].item(), args.batch_size)
        group_losses.update(loss_dic['group'].item(), args.batch_size)
        cls_losses.update(args.reg[1] * loss_dic['cls'].item(), args.batch_size)
        ent_losses.update(args.reg[0] * loss_dic['ent'].item(), args.batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train loss = protosim({:.3f}) + groupsim({:.3f}) + cls({:.3f}) + ent({:.3f})'.format(proto_losses.avg,  group_losses.avg, cls_losses.avg, ent_losses.avg))



def val(args, model, device, train_label_loader, train_unlabel_loader, epoch, n_cls, old_graph):
    model.eval()
    features_l = []
    features_u = []
    targets_l = []
    targets_u = []
    unlabel_loader_iter = cycle(train_unlabel_loader)
    with torch.no_grad():
        for batch_idx, ((x_l, _), y_l) in enumerate(train_label_loader):
            ((x_u, _), y_u) = next(unlabel_loader_iter)
            x_l, y_l, x_u, y_u = x_l.to(device), y_l.to(device), x_u.to(device), y_u.to(device)
            feature_l  = model.encoder(x_l)
            feature_u  = model.encoder(x_u)
            features_l.append(feature_l)
            features_u.append(feature_u)
            targets_l.append(y_l)
            targets_u.append(y_u)

        targets_u = torch.hstack(targets_u)
        features_l = torch.vstack(features_l)
        features_u = torch.vstack(features_u)
        targets_l = torch.hstack(targets_l)
        features = torch.cat((features_l,features_u),0)
        targets = torch.cat((targets_l,targets_u),0)

        prototypes = model.prototypes[model.proto_ind]
        prototypes = F.normalize(prototypes, dim=1)
        features = F.normalize(features, dim=1)
        features_l = F.normalize(features_l, dim=1)
        features_u = F.normalize(features_u, dim=1)
        

        dist_matrix = torch.mm(features, prototypes.t())
        edge_graph = proto_graph(dist_matrix, args.nn)

        ind = edge_graph.diagonal() >= args.min_count
        edge_graph = edge_graph[ind,:][:,ind]

        dist_matrix = dist_matrix[:,ind]
        proto_ind = model.proto_ind.clone()
        proto_ind[proto_ind==True] = ind
        prototypes = prototypes[ind]
        
        seed = 0
        edge_graph = reknn_graph(dist_matrix, args.nn, mode='min')

        targets_l = targets_l.cpu().numpy()

        def group_discovery(edge_graph, prototypes, args, targets_l, eps, n_cls=n_cls):
            proto_label, proto_mask = graph_cluster(edge_graph, prototypes, lamda=args.lamda_graph, method=args.group_method, seed=seed, n_cls=n_cls, eps=eps)
            group_label, group_index = np.unique(proto_label, return_index=True)
            group_mask = proto_mask[group_index]

            # match y_l to proto group label
            preds_proto = np.argmax(dist_matrix.cpu().numpy(),1)
            preds_group = proto_label[preds_proto]

            
            preds_group_l = preds_group[:len(features_l)]
            contingency_matrix = metrics.cluster.contingency_matrix(targets_l, preds_group_l)
            target_ind, group_label_new_l = linear_sum_assignment(contingency_matrix.max() - contingency_matrix)
            
            group_label_new = np.r_[group_label[group_label_new_l], np.delete(group_label, group_label_new_l, axis=0)]
            group_mask_new = group_mask[group_label_new]

            mp = group_label.copy()
            mp[group_label_new] = group_label
            pred_ordered = mp[preds_group_l]
            acc = accuracy(pred_ordered, targets_l)
            return acc,  preds_proto, proto_mask, group_mask, group_mask_new
        
        acc_best = 0
        eps_best = args.eps
        n_cls_best = n_cls


        if args.unknown_n_cls:
            if args.group_method in ['propagation', 'connected'] and args.dataset=='cifar10':
                interval = (0.5, 0.99, 0.02)
            elif args.group_method in ['louvain'] and args.dataset=='cifar10':
                interval = (2, 5, 0.2)
            elif args.group_method in ['louvain'] and args.dataset=='cifar100':
                interval = (6, 12, 0.2)
            else:
                raise Exception("Invalid clustering method. ['propagation', 'connected', 'louvain'] for cifar10, ['louvain'] for cifar100.")
            for eps_t in np.arange(*interval):
                acc, _, _, group_mask, _ = group_discovery(edge_graph, prototypes, args, targets_l, eps_t)
                print('EPS:{:.2f}, NUM:{:d}, ACC:{:.4f}'.format(eps_t, group_mask.shape[0], acc))
                if acc > acc_best:
                    acc_best = acc
                    eps_best = eps_t
            print('** Best Group_num {} **'.format(model.group_mask.shape[0]))

        acc,  preds_proto, proto_mask, group_mask, group_mask_new = group_discovery(edge_graph, prototypes, args, targets_l, eps_best, n_cls_best)
        return edge_graph, proto_mask, group_mask_new, proto_ind, acc


def test(args, model, labeled_num, device, test_loader, epoch):
    model.eval()
    preds = np.array([])
    features = []
    targets = np.array([])
    confs = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            pred, conf, feature = model.pred(x)

            features.append(feature)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())

    features = torch.cat(features, 0)
    targets = targets.astype(int)
    preds = preds.astype(int)
    features = features.cpu().numpy()


    known_mask = targets < labeled_num
    unknown_mask = ~known_mask

    all_acc = cluster_acc(preds, targets)
    known_acc = accuracy(preds[known_mask], targets[known_mask])
    unknown_acc = cluster_acc(preds[unknown_mask], targets[unknown_mask])

    print('Test All ACC {:.4f}, Known ACC {:.4f}, Unknown ACC {:.4f}'.format(all_acc, known_acc, unknown_acc))


    
    


def main():
    parser = argparse.ArgumentParser(description='OpenNCD')
    parser.add_argument('--milestones', nargs='+', type=int, default=[90, 120])
    parser.add_argument('--dataset', default='cifar10', help='dataset')
    parser.add_argument('--backbone', default='resnet18', help='backbone setting')
    parser.add_argument('--labeled_num', default=5, type=int)
    parser.add_argument('--labeled_ratio', default=0.1, type=float)
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--exp_root', type=str, default='./results/')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('-b', '--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--lamda', default=1, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--ldim', default=32, type=int)
    parser.add_argument('--nn', default=3, type=int)
    parser.add_argument('--lamda_graph', default=1, type=float)
    parser.add_argument('--tau', default=0.1, type=float)
    parser.add_argument('--min_count', default=0, type=int)
    parser.add_argument('--reg', type=float, nargs='+', default=[1, 1], help='loss weights')
    parser.add_argument('--init', default=False, action='store_true')
    parser.add_argument('--lr', default=2e-3, type=float, help='learning rate of backbone')
    parser.add_argument('--lr_proto', default=2e-3, type=float, help='learning rate of prototypes')
    parser.add_argument('--n_proto', default=50, type=int, help='number of prototypes')
    parser.add_argument('--eps', default=0.7, type=float, help='grouping threshold')
    parser.add_argument('--warm_epoch', default=5, type=float, help='warm up epoch')
    parser.add_argument('--fix_epoch', default=20, type=float, help='grouping epoch')
    parser.add_argument('--group_method', default='spectral', help='[louvain/connected/propagation] if unknown_n_cls')
    parser.add_argument('--unknown_n_cls', action='store_true', help='action if n_classes is unknown')
    parser.add_argument('--save_log', action='store_true', help='action to save output')
    
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = '%s' %args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current Device:', device)

    setup_seed(args.seed)
    args.time_stamp = time.strftime("%y%m%d_%H%M%S", time.localtime())

    print('Current Time: ', args.time_stamp)
    print(args)


    root = './data'
    if args.dataset == 'cifar10':
        train_label_set = datasets.OPENWORLDCIFAR10(root=root, labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar10_train']))
        train_unlabel_set = datasets.OPENWORLDCIFAR10(root=root, labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar10_train']), unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR10(root=root, labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=datasets.dict_transform['cifar10_test'], unlabeled_idxs=train_label_set.unlabeled_idxs)
        n_classes = 10
        fix_epoch = 70
        warm_up_epoch = args.warm_epoch

    elif args.dataset == 'cifar100':
        args.labeled_num = 50
        train_label_set = datasets.OPENWORLDCIFAR100(root=root, labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar100_train']))
        train_unlabel_set = datasets.OPENWORLDCIFAR100(root=root, labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar100_train']), unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR100(root=root, labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=datasets.dict_transform['cifar100_test'], unlabeled_idxs=train_label_set.unlabeled_idxs)
        args.n_proto = 500
        n_classes = 100
        args.lamda_graph = 0.9
        args.nn = 10
        fix_epoch = 70
        warm_up_epoch = args.warm_epoch

    else:
        warnings.warn('Dataset is not listed')
        return

    if not args.unknown_n_cls:
        args.group_method = 'spectral'
        fix_epoch = 20


    NAME = '{}_lratio0{:d}{}'.format(args.dataset, int(args.labeled_ratio*10), 
                                           "_"+args.group_method if args.unknown_n_cls else '')
    args.savedir = args.exp_root + NAME
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    if args.save_log:
        sys.stdout = Logger(args.savedir + '/{}.log'.format(NAME), sys.stdout)


    labeled_len = len(train_label_set)
    unlabeled_len = len(train_unlabel_set)
    labeled_batch_size = int(args.batch_size * labeled_len / (labeled_len + unlabeled_len))

    # Initialize the splits
    train_label_loader = torch.utils.data.DataLoader(train_label_set, batch_size=labeled_batch_size, shuffle=True, num_workers=8, drop_last=True)
    train_unlabel_loader = torch.utils.data.DataLoader(train_unlabel_set, batch_size=args.batch_size - labeled_batch_size, shuffle=True, num_workers=8, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=8)

    model = Model(arch=args.backbone, proto_num=args.n_proto,  latent_dim=args.ldim, tau=args.tau, device=device)
    model = model.to(device)

    if args.dataset == 'cifar10':
        state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
    elif args.dataset == 'cifar100':
        state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')

    model.encoder.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    # Freeze the earlier filters
    for name, param in model.encoder.named_parameters():
        if 'projector' not in name and 'layer4' not in name:
            param.requires_grad = False

    encoder_params = list(model.encoder.parameters())
    prototype_params = [model.prototypes]
    optimizer = optim.Adam([{'params': filter(lambda p: p.requires_grad, encoder_params)},
                            {'params': prototype_params,'lr': args.lr_proto}], lr=args.lr, betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    start_epoch = 0
    acc_t = 0

    for epoch in range(start_epoch, args.epochs):
        if epoch < warm_up_epoch: # warm up
            print('*****Epoch {} Warming stage*****'.format(epoch))
            args.reg[1] = 0
        elif epoch < fix_epoch:
            print('*****Epoch {} Grouping stage*****'.format(epoch))
            args.reg[1] = 0.1 if args.unknown_n_cls else 1
            proto_graph, proto_mask, group_mask, proto_ind, acc = val(args, model, device, train_label_loader, train_unlabel_loader, epoch, n_classes, model.proto_mask)
            if acc > acc_t and epoch % 2 == 1: # fix 
                model.proto_graph = proto_graph.to(device)
                model.proto_mask = torch.tensor(proto_mask).to(device)
                model.group_mask = torch.tensor(group_mask).to(device)
                model.proto_ind = proto_ind
                acc_t = acc if not epoch == fix_epoch else 0       
        else:
            print('*****Epoch {} Fixing stage*****'.format(epoch)) 
            args.reg[1] = 1
            pass


        train(args, model, device, train_label_loader, train_unlabel_loader, optimizer, epoch)
        test(args, model, args.labeled_num, device, test_loader, epoch)
        scheduler.step()

    print('Current Time: ', args.time_stamp)
    torch.save(model.state_dict(), args.savedir + '/ckpt.pth')

if __name__ == '__main__':
    main()
