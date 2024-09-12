# -*- coding: utf-8 -*-
# @Time   : 2023/4/21 12:00
# @Author : Zhen Tian
# @Email  : chenyuwuxinn@gmail.com
# @File   : eulernet.py

r"""
EulerNet
################################################
Reference:
    Zhen Tian et al. "EulerNet: Adaptive Feature Interaction Learning via Euler's Formula for CTR Prediction." in SIGIR 2023.

Reference code:
    https://github.com/chenyuwuxin/EulerNet

"""

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.init import xavier_normal_, constant_
from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.loss import RegLoss, BPRLoss


class EulerNet(ContextRecommender):
    r"""EulerNet is a context-based recommendation model.
    It can adaptively learn the arbitrary-order feature interactions in a complex vector space
    by conducting space mapping according to Euler's formula. Meanwhile, it can jointly capture
    the explicit and implicit feature interactions in a unified model architecture.
    """

    def __init__(self, config, dataset):
        super(EulerNet, self).__init__(config, dataset)
        field_num = self.field_num = self.num_feature_field
        shape_list = [config.embedding_size * field_num] + [
            num_neurons * config.embedding_size for num_neurons in config.order_list
        ]

        interaction_shapes = []
        for inshape, outshape in zip(shape_list[:-1], shape_list[1:]):
            interaction_shapes.append(EulerInteractionLayer(config, inshape, outshape))

        self.Euler_interaction_layers = nn.Sequential(*interaction_shapes)
        self.mu = nn.Parameter(torch.ones(1, field_num, 1))
        if config['multi_cls']:
            self.reg = nn.Linear(shape_list[-1], config['multi_cls'])
            self.ce_loss = nn.CrossEntropyLoss()
            self.multi_cls = True
            self.multi_ratio = config['multi_ratio']
            self.softmax = nn.Softmax(dim=1)
            self.threshold = config['threshold']['rating']
            self.pred_all_probs = config['pred_all_probs']
        else:
            self.reg = nn.Linear(shape_list[-1], 1)
            self.loss = nn.BCEWithLogitsLoss()
            self.multi_cls = False
            self.sigmoid = nn.Sigmoid()
        self.reg_weight = config.reg_weight
        nn.init.normal_(self.reg.weight, mean=0, std=0.01)
        
        self.reg_loss = RegLoss()
        self.apply(self._init_other_weights)
        self.config = config
        if config['add_ranking_loss']:
            self.ranking_ratio = config['ranking_ratio']
            if config['add_ranking_loss'] == 'both':
                self.pos_ratio = config['pos_ratio']
                assert self.pos_ratio >= 0 and self.pos_ratio <= 1
        self.mf_loss = BPRLoss()

    def _init_other_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, interaction):
        fm_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]
        r, p = self.mu * torch.cos(fm_all_embeddings), self.mu * torch.sin(
            fm_all_embeddings
        )
        o_r, o_p = self.Euler_interaction_layers((r, p))
        o_r, o_p = o_r.reshape(o_r.shape[0], -1), o_p.reshape(o_p.shape[0], -1)
        re, im = self.reg(o_r), self.reg(o_p)
        logits = re + im
        return logits.squeeze(-1)
    
    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        if self.multi_cls:
            ratings = interaction['rating'].long() - 1
            rating_loss = self.ce_loss(output, ratings)
            neg_output = output[:, :self.threshold - 1]
            pos_output = output[:, self.threshold - 1:]
            neg_output = neg_output.sum(dim=1, keepdim=True)
            pos_output = pos_output.sum(dim=1, keepdim=True)
            label_output = torch.cat([neg_output, pos_output], dim=1)
            label_loss = self.ce_loss(label_output, label.long())
            # return label_loss + self.RegularLoss(self.reg_weight), rating_loss * self.multi_ratio
            return (1 - self.multi_ratio) * (label_loss + self.RegularLoss(self.reg_weight)), rating_loss * self.multi_ratio
        else:
            return self.loss(output, label) + self.RegularLoss(self.reg_weight)

    def predict(self, interaction):
        output = self.forward(interaction)
        if self.multi_cls:
            if self.pred_all_probs:
                return self.softmax(output)
            neg_output = output[:, :self.threshold - 1]
            pos_output = output[:, self.threshold - 1:]
            neg_output = neg_output.sum(dim=1, keepdim=True)
            pos_output = pos_output.sum(dim=1, keepdim=True)
            label_output = torch.cat([neg_output, pos_output], dim=1)
            return self.softmax(label_output)[:, 1]
        return self.sigmoid(output)

    def RegularLoss(self, weight):
        if weight == 0:
            return 0
        loss = 0
        for _ in ["Euler_interaction_layers", "mu", "reg"]:
            comp = getattr(self, _)
            if isinstance(comp, nn.Parameter):
                loss += torch.norm(comp, p=2)
                continue
            for params in comp.parameters():
                loss += torch.norm(params, p=2)
        return loss * weight


class EulerInteractionLayer(nn.Module):
    r"""Euler interaction layer is the core component of EulerNet,
    which enables the adaptive learning of explicit feature interactions. An Euler
    interaction layer performs the feature interaction under the complex space one time,
    taking as input a complex representation and outputting a transformed complex representation.
    """

    def __init__(self, config, inshape, outshape):
        super().__init__()
        self.feature_dim = config.embedding_size
        self.apply_norm = config.apply_norm

        init_orders = torch.softmax(
            torch.randn(inshape // self.feature_dim, outshape // self.feature_dim)
            / 0.01,
            dim=0,
        )
        self.inter_orders = nn.Parameter(init_orders)
        self.im = nn.Linear(inshape, outshape)

        self.bias_lam = nn.Parameter(
            torch.randn(1, self.feature_dim, outshape // self.feature_dim) * 0.01
        )
        self.bias_theta = nn.Parameter(
            torch.randn(1, self.feature_dim, outshape // self.feature_dim) * 0.01
        )
        nn.init.normal_(self.im.weight, mean=0, std=0.1)

        self.drop_ex = nn.Dropout(p=config.drop_ex)
        self.drop_im = nn.Dropout(p=config.drop_im)
        self.norm_r = nn.LayerNorm([self.feature_dim])
        self.norm_p = nn.LayerNorm([self.feature_dim])

    def forward(self, complex_features):
        r, p = complex_features

        lam = r**2 + p**2 + 1e-8
        theta = torch.atan2(p, r)
        lam, theta = lam.reshape(lam.shape[0], -1, self.feature_dim), theta.reshape(
            theta.shape[0], -1, self.feature_dim
        )
        r, p = self.drop_im(r), self.drop_im(p)

        lam = 0.5 * torch.log(lam)
        lam, theta = torch.transpose(lam, -2, -1), torch.transpose(theta, -2, -1)
        lam, theta = self.drop_ex(lam), self.drop_ex(theta)
        lam, theta = (
            lam @ (self.inter_orders) + self.bias_lam,
            theta @ (self.inter_orders) + self.bias_theta,
        )
        lam = torch.exp(lam)
        lam, theta = torch.transpose(lam, -2, -1), torch.transpose(theta, -2, -1)

        r, p = r.reshape(r.shape[0], -1), p.reshape(p.shape[0], -1)
        r, p = self.im(r), self.im(p)
        r, p = torch.relu(r), torch.relu(p)
        r, p = r.reshape(r.shape[0], -1, self.feature_dim), p.reshape(
            p.shape[0], -1, self.feature_dim
        )

        o_r, o_p = r + lam * torch.cos(theta), p + lam * torch.sin(theta)
        o_r, o_p = o_r.reshape(o_r.shape[0], -1, self.feature_dim), o_p.reshape(
            o_p.shape[0], -1, self.feature_dim
        )
        if self.apply_norm:
            o_r, o_p = self.norm_r(o_r), self.norm_p(o_p)
        return o_r, o_p
