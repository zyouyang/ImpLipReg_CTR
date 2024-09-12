# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2020/9/16, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com

r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import time

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class LightGCN(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )

        # self.embedding_mlp = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.norm_adj_matrix.requires_grad_(True)
        
        # define gumbel probs
        self.gb = config['gb_prob']
        self.gb_train = config['gb_train']
        if self.gb:
            self.user_gb = torch.nn.Parameter(torch.rand(self.n_users, 1, 2))
            self.item_gb = torch.nn.Parameter(torch.rand(1, self.n_items, 2))
            self.user_gb.requires_grad_(True)
            self.item_gb.requires_grad_(True)
            self.gb_probs = torch.nn.Parameter(torch.rand(self.n_users, self.n_items, 2))
            self.gb_probs.requires_grad_(True)
            self.torch_interaction_matix = dataset.inter_matrix(form="torch_sparse").to(self.device)
            self.torch_interaction_matix = self.torch_interaction_matix.to_dense()

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_gb_modified_norm_adj_mat(self, tau=1.0):
        gb_probs = self.user_gb * self.item_gb # shape: (n_users, n_items, 2)
        gumbels = (
            -torch.empty_like(gb_probs, 
            memory_format=torch.legacy_contiguous_format).exponential_().log())
        gumbels = (gb_probs + gumbels) / tau
        y_soft_pre = gumbels.softmax(dim=-1)
        y_soft, y_hard = y_soft_pre.max(-1)
        # y_soft = torch.sigmoid(gumbels)
        # y_hard = (y_soft > 0.5).int()
        flip_adj = y_hard - y_soft.detach() + y_soft
        # flip_adj = F.gumbel_softmax(self.gb_probs, tau=tau, hard=True)
        ui_adj = self.torch_interaction_matix + flip_adj - 2 * (self.torch_interaction_matix * flip_adj)

        # build adj matrix
        A = torch.zeros(self.n_users + self.n_items, self.n_users + self.n_items, device=self.device)
        A[:self.n_users, self.n_users:] = ui_adj
        A[self.n_users:, :self.n_users] = ui_adj.t()

        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = sumArr.flatten() + 1e-7
        diag = torch.pow(diag, -0.5)
        D = torch.diag(diag)
        # L = D * A * D
        L = torch.mm(torch.mm(D, A), D)
        L = L.to_sparse()
        return L

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, epoch_idx=None, norm_adj_matrix=None):
        ego_embeddings = self.get_ego_embeddings()
        ego_user_embeddings, ego_item_embeddings = torch.split(
            ego_embeddings, [self.n_users, self.n_items]
        )
        all_embeddings = ego_embeddings
        embeddings_list = [all_embeddings]
        norm_adj_matrix = self.norm_adj_matrix

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings, ego_user_embeddings, ego_item_embeddings

    def calculate_loss(self, interaction, epoch_idx=None):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, ego_user_embeddings, ego_item_embeddings = self.forward(epoch_idx)   
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss
        if self.gb_train:
            lip_mat_user, lip_mat_item = [], []
            # deep network
            for i in range(item_all_embeddings.shape[1]):
                v = torch.zeros_like(item_all_embeddings)
                v[:, i] = 1
                gradients = autograd.grad(outputs=item_all_embeddings, inputs=self.norm_adj_matrix.to_dense(), grad_outputs=v,
                                        create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]
                breakpoint()
                grad_norm = torch.norm(gradients, dim=1).unsqueeze(dim=1)
                lip_mat_item.append(grad_norm)

                v = torch.zeros_like(user_all_embeddings)
                v[:, i] = 1
                gradients = autograd.grad(outputs=user_all_embeddings, inputs=ego_user_embeddings, grad_outputs=v,
                                        create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]
                grad_norm = torch.norm(gradients, dim=1).unsqueeze(dim=1)
                lip_mat_user.append(grad_norm)

            lip_concat_user = torch.cat(lip_mat_user, dim=1)
            lip_con_norm_user = torch.norm(lip_concat_user, dim=1)
            lip_loss_user = torch.max(lip_con_norm_user)
            lip_concat_item = torch.cat(lip_mat_item, dim=1)
            lip_con_norm_item = torch.norm(lip_concat_item, dim=1)
            lip_loss_item = torch.max(lip_con_norm_item)
            lip_loss = (lip_loss_item + lip_loss_user) / 2
            loss = loss + 0.1 * lip_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, _, _ = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _, _ = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
