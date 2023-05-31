import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
import sys
import os.path
import torch.nn.functional as F
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.cluster import SpectralClustering, AffinityPropagation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import networkx as nx
import community

plt.rcParams.update({'figure.max_open_warning': 0})

class TransformTwice:
    # two different random transform with one image
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def accuracy(output, target):
    
    num_correct = np.sum(output == target)
    res = num_correct / len(target)

    return res


def cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_pred.size


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def extract_features(model, data_loader):
    # extract features from dataloader to numpy array
    features = []
    targets = []
    model.eval()
    with  torch.no_grad():
        for (image, image2), label in data_loader:
            image= image.cuda()
            feature = model.encoder(image)
            features.append(feature)
            targets.append(label)
    features = torch.cat(features, 0).detach().cpu().numpy()
    targets = torch.cat(targets, 0).detach().numpy()
    return features, targets


def topK(self, matrix, k, axis=-1):
    if k > 0:
        topK_ind = np.argpartition(matrix, kth=-k, axis=axis)[:,-k:]
    else: # bottom K
        topK_ind = np.argpartition(matrix, kth=-k, axis=axis)[:,:-k]
    topK_elements = np.take_along_axis(matrix, topK_ind, axis=axis) 
    return topK_elements


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    purity_pred = np.amax(contingency_matrix, axis=0) / np.sum(contingency_matrix, axis=0)
    purity_score = np.mean(purity_pred)
    return purity_pred, purity_score


def proto_graph(dist_matrix, n_nearest_neighbor):
    n_topk = n_nearest_neighbor + 1
    n_protos = dist_matrix.shape[1]
    _, topk_inds = dist_matrix.topk(n_topk, dim=1, largest=True, sorted=True)

    edges_pairs = [topk_inds[:,[0,k]] for k in range(n_topk)]
    edges = torch.vstack(edges_pairs)

    weights = torch.ones_like(edges[:,0])

    graph = torch.sparse_coo_tensor(edges.t(), weights, size=(n_protos, n_protos))
    graph = graph.to_dense().type(torch.Tensor)
    
    return graph

def graph_cluster(edge_graph, prototypes, lamda=0.5, method='spectral', seed=0, n_cls=10, eps=0.7):

    edge_graph = edge_graph / torch.clamp(edge_graph.diag().reshape(-1,1), 1)
    edge_graph = edge_graph.cpu().numpy()
    edge_graph = (edge_graph + edge_graph.T) / 2 - np.diag(edge_graph.diagonal())
    attr_graph = ((torch.mm(prototypes , prototypes.T) + 1) / 2).cpu().numpy()

    graph = lamda * edge_graph + (1-lamda) * attr_graph
    
    if method == 'spectral': # for known cluster number
        clu_spec = SpectralClustering(n_clusters=n_cls,
                        assign_labels='kmeans',
                        random_state=seed, affinity='precomputed').fit(graph)
        label = clu_spec.labels_
    elif method == 'propagation': # for unknown cluster number
        clu_affi = AffinityPropagation(affinity='precomputed', 
                        damping=eps, 
                        verbose=False).fit(graph)
        label = clu_affi.labels_
    elif method == 'connected': # for unknown cluster number
        select_graph = csr_matrix((graph>eps))
        n_components, label = connected_components(csgraph=select_graph, directed=False, return_labels=True)
    elif method == 'louvain': # for unknown cluster number
        partition = community.best_partition(nx.from_numpy_array(graph), resolution=eps)
        label = np.fromiter(partition.values(), dtype=int)

    mask = graph.copy()
    for i in range(len(mask)):
        for j in range(len(mask)):
            if label[i] == label[j]:
                mask[i,j] = 1
            else:
                mask[i,j] = 0

    return label, mask

def reknn_graph(dist_matrix, n_nearest_neighbor, mode='harmonic_mean'):
    # n_topk = n_nearest_neighbor 
    _, top_indices = torch.topk(dist_matrix, n_nearest_neighbor)
    onehot = torch.zeros_like(dist_matrix).scatter(1, top_indices, 1).bool()
    jaccard = torch.zeros(onehot.shape[1], onehot.shape[1])
    ind_reknn = onehot.t()
    num_reknn = onehot.t().sum(1)
    for i in range(onehot.shape[1]):
        for j in range(onehot.shape[1]):
            n_intersection = (onehot.t()[i] & onehot.t()[j]).sum()
            if n_intersection == 0:
                jaccard[i][j] = 0
                continue
            if mode == 'jaccard':
                n_union = (onehot.t()[i] | onehot.t()[j]).sum()
                jaccard[i][j] = n_intersection / n_union
            if mode == 'harmonic_mean':
                jaccard[i][j] = n_intersection / 2 * num_reknn[i] * num_reknn[j] / (num_reknn[i] + num_reknn[j])
            elif mode == 'geometric_mean':
                jaccard[i][j] = n_intersection / torch.sqrt(num_reknn[i] * num_reknn[j])
            elif mode == 'min':
                jaccard[i][j] = n_intersection / torch.min(num_reknn[i], num_reknn[j])
            elif mode == 'max':
                jaccard[i][j] = n_intersection / torch.max(num_reknn[i], num_reknn[j])
    return jaccard


# def proto_label_graph(features_l, targets_l, prototypes):
#     dist_matrix_l = torch.mm(features_l, prototypes.t())
#     proto_len = prototypes.shape[0]
#     proto_count_l = torch.mm(F.one_hot(dist_matrix_l.max(1)[1], num_classes=proto_len).T.float(),  F.one_hot(targets_l).float())
#     proto_graph_l = torch.zeros(proto_len, proto_len)

#     min_count = targets_l.shape[0] / proto_len
#     for i in range(proto_len):
#         for j in range(proto_len):
#             if proto_count_l[i].sum() > min_count and proto_count_l[j].sum() > min_count:
#                 if proto_count_l[i].argmax() == proto_count_l[j].argmax():
#                     proto_graph_l[i][j] = 1
#                 else:
#                     proto_graph_l[i][j] = -1
#             else:
#                 proto_graph_l[i][j] = 0
#     return proto_graph_l


