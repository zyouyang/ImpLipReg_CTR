# -*- coding: utf-8 -*-
# @Time   : 2020/08/30
# @Author : Xinyan Fan
# @Email  : xinyan.fan@ruc.edu.cn
# @File   : widedeep.py

r"""
WideDeep
#####################################################
Reference:
    Heng-Tze Cheng et al. "Wide & Deep Learning for Recommender Systems." in RecSys 2016.
"""
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import MLPLayers
from recbole.model.loss import BPRLoss


class WideDeep(ContextRecommender):
    r"""WideDeep is a context-based recommendation model.
    It jointly trains wide linear models and deep neural networks to combine the benefits
    of memorization and generalization for recommender systems. The wide component is a generalized linear model
    of the form :math:`y = w^Tx + b`. The deep component is a feed-forward neural network. The wide component
    and deep component are combined using a weighted sum of their output log odds as the prediction,
    which is then fed to one common logistic loss function for joint training.
    """

    def __init__(self, config, dataset):
        super(WideDeep, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]
        self.config = config
        if config['multi_cls']:
            self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], config['multi_cls'])
            self.ce_loss = nn.CrossEntropyLoss()
            self.multi_cls = True
            self.multi_ratio = config['multi_ratio']
            self.softmax = nn.Softmax(dim=1)
            self.threshold = config['threshold']['rating']
            self.pred_all_probs = config['pred_all_probs']
        else:
            self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
            self.loss = nn.BCEWithLogitsLoss()
            self.multi_cls = False
            self.sigmoid = nn.Sigmoid()

        # define layers and loss
        size_list = [
            self.embedding_size * self.num_feature_field
        ] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)

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
        widedeep_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]
        batch_size = widedeep_all_embeddings.shape[0]
        fm_output = self.first_order_linear(interaction)

        mlp_output = self.mlp_layers(widedeep_all_embeddings.view(batch_size, -1))
        deep_output = self.deep_predict_layer(mlp_output)

        output = fm_output + deep_output
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
            # return label_loss, rating_loss * self.multi_ratio
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

