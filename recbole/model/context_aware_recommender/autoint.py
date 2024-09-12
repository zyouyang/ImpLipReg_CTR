# -*- coding: utf-8 -*-
# @Time   : 2020/09/01
# @Author : Shuqing Bian
# @Email  : shuqingbian@gmail.com
# @File   : autoint.py

r"""
AutoInt
################################################
Reference:
    Weiping Song et al. "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"
    in CIKM 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_
import torch.autograd as autograd

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import MLPLayers
from recbole.model.loss import BPRLoss


class AutoInt(ContextRecommender):
    """AutoInt is a novel CTR prediction model based on self-attention mechanism,
    which can automatically learn high-order feature interactions in an explicit fashion.

    """

    def __init__(self, config, dataset):
        super(AutoInt, self).__init__(config, dataset)

        # load parameters info
        self.attention_size = config["attention_size"]
        self.dropout_probs = config["dropout_probs"]
        self.n_layers = config["n_layers"]
        self.num_heads = config["num_heads"]
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.has_residual = config["has_residual"]
        self.config = config
        self.atten_output_dim = self.num_feature_field * self.attention_size
        if config['multi_cls']:
            self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], config['multi_cls'])
            self.attn_fc = torch.nn.Linear(self.atten_output_dim, config['multi_cls'])
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
            self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)

        # define layers and loss
        self.att_embedding = nn.Linear(self.embedding_size, self.attention_size)
        self.embed_output_dim = self.num_feature_field * self.embedding_size
        size_list = [self.embed_output_dim] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(size_list, dropout=self.dropout_probs[1])
        # multi-head self-attention network
        self.self_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.attention_size, self.num_heads, dropout=self.dropout_probs[0]
                )
                for _ in range(self.n_layers)
            ]
        )
        
        if self.has_residual:
            self.v_res_embedding = torch.nn.Linear(
                self.embedding_size, self.attention_size
            )

        self.dropout_layer = nn.Dropout(p=self.dropout_probs[2])

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def autoint_layer(self, infeature):
        """Get the attention-based feature interaction score

        Args:
            infeature (torch.FloatTensor): input feature embedding tensor. shape of[batch_size,field_size,embed_dim].

        Returns:
            torch.FloatTensor: Result of score. shape of [batch_size,1] .
        """

        att_infeature = self.att_embedding(infeature)
        cross_term = att_infeature.transpose(0, 1)
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        cross_term = cross_term.transpose(0, 1)
        # Residual connection
        if self.has_residual:
            v_res = self.v_res_embedding(infeature)
            cross_term += v_res
        # Interacting layer
        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)
        batch_size = infeature.shape[0]
        mlp_output = self.mlp_layers(infeature.view(batch_size, -1))
        att_output = self.attn_fc(cross_term) + self.deep_predict_layer(mlp_output)
        return att_output

    def forward(self, interaction):
        autoint_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]
        fol_output = self.first_order_linear(interaction)
        autoint_output = self.autoint_layer(autoint_all_embeddings)
        output = fol_output + autoint_output
        return output.squeeze(1)

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


    # def calculate_loss(self, interaction):
    #     label = interaction[self.LABEL]
    #     ratings = interaction['rating']
    #     output = self.forward(interaction)

    #     # label == 1: existing interaction
    #     # label == 0: negative sampling
    #     rating_label = ratings >= self.config['threshold']['rating']
    #     rating_label = rating_label.float()
    #     true_inds = label == 1
    #     rating_loss = self.loss(output[true_inds], rating_label[true_inds])
    #     rating_loss = rating_loss.mean()

    #     if self.config['add_ranking_loss']:
    #         pos_inds = torch.logical_and(label == 1, ratings >= self.config['threshold']['rating'])
    #         neg_inds = torch.logical_and(label == 0, ratings >= self.config['threshold']['rating'])
    #         pos_output = output[pos_inds]
    #         pos_output = output[pos_inds]
    #         neg_output = output[neg_inds]
    #         ranking_loss_pos = self.mf_loss(pos_output, neg_output)

    #         if self.config['both_ranking']:
    #             neg_inds = torch.logical_and(label == 1, ratings <= 1)
    #             pos_inds = torch.logical_and(label == 0, ratings <= 1)
    #             pos_output = output[pos_inds]
    #             neg_output = output[neg_inds]
    #             ranking_loss_neg = self.mf_loss(pos_output, neg_output)
    #             ranking_loss = (ranking_loss_pos + ranking_loss_neg) / 2
    #         else:
    #             ranking_loss = ranking_loss_pos

    #         return (1 - self.ranking_loss_ratio) * rating_loss, self.ranking_loss_ratio * ranking_loss
        
    #     return rating_loss

    # def predict(self, interaction):
    #     output = self.forward(interaction)
    #     probs = self.sigmoid(output)
    #     return probs