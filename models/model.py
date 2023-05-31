import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import resnet18


class Model(nn.Module):
    def __init__(self, arch='resnet18', proto_num=50, latent_dim=32, tau=0.1, init=False, device='cuda'):
        super(Model, self).__init__()
        self.arch = arch
        self.proto_num = proto_num
        self.latent_dim = latent_dim
        self.tau = tau
        self.encoder = resnet18(num_classes=latent_dim)

        self.prototypes = nn.Parameter(torch.randn(self.proto_num, self.latent_dim).to(device), requires_grad=True) 
        self.proto_ind = torch.ones(len(self.prototypes), dtype=torch.bool)
        self.group_mask = torch.eye(len(self.prototypes)).to(device)
        self.proto_graph = torch.eye(self.proto_num).to(device)
        self.proto_mask = torch.eye(self.proto_num).to(device)

        if init:
            self.prototypes.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def find_pairs(self, feature, y_l):
        feat_detach = feature.detach().flatten(start_dim=1)
        feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
        cosine_dist = torch.mm(feat_norm, feat_norm.t()) 
        labeled_len = len(y_l)
        pos_pairs = []
        target_np = y_l.cpu().numpy()
        
        for i in range(labeled_len):
            target_i = target_np[i]
            idxs = np.where(target_np == target_i)[0]
            if len(idxs) == 1:
                pos_pairs.append(idxs[0])
            else:
                selec_idx = np.random.choice(idxs, 1)  # choose one pos feature
                while selec_idx == i:                  # cannot be itself
                    selec_idx = np.random.choice(idxs, 1)
                pos_pairs.append(int(selec_idx))
          
        unlabel_cosine_dist = cosine_dist[labeled_len:, :]
        vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)  # choose the nearest feature
        pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
        pos_pairs.extend(pos_idx)
        return pos_pairs     



    def loss(self, x_l, x_l2, y_l, x_u, x_u2):
        labeled_len = len(y_l)
        
        c = self.prototypes
        c = F.normalize(c, dim=1)

        x = torch.cat([x_l, x_u], 0)
        z = self.encoder(x)
        z = F.normalize(z, dim=1)
        p = F.softmax(torch.mm(z, c.t()) / self.tau, dim=1)
        q = torch.mm(p, self.group_mask.t()) # group_mask M ~ [N_group, N_proto]  M_ij = 1 means Group_i contain Prototype_j 

        # another view
        x2 = torch.cat([x_l2, x_u2], 0)
        z2 = self.encoder(x2)
        z2 = F.normalize(z2, dim=1)
        p2 = F.softmax(torch.mm(z2, c.t()) / self.tau, dim=1)
        q2 = torch.mm(p2, self.group_mask.t()) # sum after exp

        ############ Prototype Similarity ############
        pair_ind = self.find_pairs(z, y_l) # index
        p2_pair = p2[pair_ind, :]         # pospair features of x_2
        proto_sim = - torch.log((p * p2_pair).sum(1)).mean()

        ############ Group Similarity ############
        group_sim_1 = torch.mean(torch.sum(- q2 * torch.log(q + 1e-8), dim=1)) 
        group_sim_2 = torch.mean(torch.sum(- q * torch.log(q2 + 1e-8), dim=1))
        group_sim = 0.5 * group_sim_1 + 0.5 * group_sim_2

        ############ Multi-prototype CE ############
        y_l_onehot = F.one_hot(y_l, len(self.group_mask))
        cls_loss = torch.sum(- y_l_onehot * torch.log(q[:labeled_len] + 1e-8)) / len(y_l)

        ############ Entropy Regularization ############
        p_prior = F.normalize(F.normalize(self.group_mask, p=1, dim=1).sum(0), p=1, dim=0)
        p_proto = p.mean(0)
        ent_loss = torch.sum(p_proto * torch.log(p_proto / p_prior) + 1e-7) * 10

        loss = {'proto': proto_sim, 'group': group_sim, 'cls': cls_loss, 'ent': ent_loss}
 
        return loss


    def pred(self, x):
        z = self.encoder(x)
        c = self.prototypes
        z = F.normalize(z, dim=1)
        c = F.normalize(c, dim=1)
        dist = torch.mm(z, c.t())

        p = F.softmax(dist / self.tau, dim=1)
        conf, pred_proto = torch.max(p, axis=1)
        pred_onehot = torch.mm(F.one_hot(pred_proto, len(c)).float(), self.group_mask.t())
        pred = torch.argmax(pred_onehot, dim=1)
        
        return pred, conf, z





