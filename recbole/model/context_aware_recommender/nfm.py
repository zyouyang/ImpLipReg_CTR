# -*- coding: utf-8 -*-
# @Time   : 2020/7/14
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.com
# @File   : nfm.py

r"""
NFM
################################################
Reference:
    He X, Chua T S. "Neural factorization machines for sparse predictive analytics" in SIGIR 2017
"""
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import BaseFactorizationMachine, MLPLayers


class NFM(ContextRecommender):
    """NFM replace the fm part as a mlp to model the feature interaction."""

    def __init__(self, config, dataset):
        super(NFM, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]

        # define layers and loss
        size_list = [self.embedding_size] + self.mlp_hidden_size
        self.fm = BaseFactorizationMachine(reduce_sum=False)
        self.bn = nn.BatchNorm1d(num_features=self.embedding_size)
        self.mlp_layers = MLPLayers(
            size_list, self.dropout_prob, activation="sigmoid", bn=True
        )
        self.config = config
        if config['multi_cls']:
            self.ce_loss = nn.CrossEntropyLoss()
            self.multi_cls = True
            self.multi_ratio = config['multi_ratio']
            self.softmax = nn.Softmax(dim=1)
            self.threshold = config['threshold']['rating']
            self.pred_all_probs = config['pred_all_probs']
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], config['multi_cls'])
        else:
            self.multi_cls = False
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1, bias=False)
            self.sigmoid = nn.Sigmoid()
            self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, interaction):
        nfm_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]
        bn_nfm_all_embeddings = self.bn(self.fm(nfm_all_embeddings))

        output = self.predict_layer(
            self.mlp_layers(bn_nfm_all_embeddings)
        ) + self.first_order_linear(interaction)
        return output.squeeze(-1)

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
            return (1 - self.multi_ratio) * label_loss, rating_loss * self.multi_ratio
        else:
            return self.loss(output, label)

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