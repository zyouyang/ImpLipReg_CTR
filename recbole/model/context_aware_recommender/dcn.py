# _*_ coding: utf-8 _*_
# @Time : 2020/10/4
# @Author : Zhichao Feng
# @Email  : fzcbupt@gmail.com

# UPDATE
# @Time   : 2020/10/21
# @Author : Zhichao Feng
# @email  : fzcbupt@gmail.com

r"""
DCN
################################################
Reference:
    Ruoxi Wang at al. "Deep & Cross Network for Ad Click Predictions." in ADKDD 2017.

Reference code:
    https://github.com/shenweichen/DeepCTR-Torch
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
import torch.autograd as autograd

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import MLPLayers
from recbole.model.loss import RegLoss


class DCN(ContextRecommender):
    """Deep & Cross Network replaces the wide part in Wide&Deep with cross network,
    automatically construct limited high-degree cross features, and learns the corresponding weights.

    """

    def __init__(self, config, dataset):
        super(DCN, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.cross_layer_num = config["cross_layer_num"]
        self.reg_weight = config["reg_weight"]
        self.dropout_prob = config["dropout_prob"]
        self.jaclip = config['jaclip']
        self.jaclip_weight = config['jaclip_weight']
        self.loss_type = config['loss_type']

        # define layers and loss
        # init weight and bias of each cross layer
        self.cross_layer_w = nn.ParameterList(
            nn.Parameter(
                torch.randn(self.num_feature_field * self.embedding_size).to(
                    self.device
                )
            )
            for _ in range(self.cross_layer_num)
        )
        self.cross_layer_b = nn.ParameterList(
            nn.Parameter(
                torch.zeros(self.num_feature_field * self.embedding_size).to(
                    self.device
                )
            )
            for _ in range(self.cross_layer_num)
        )

        # size of mlp hidden layer
        size_list = [
            self.embedding_size * self.num_feature_field
        ] + self.mlp_hidden_size
        # size of cross network output
        in_feature_num = (
            self.embedding_size * self.num_feature_field + self.mlp_hidden_size[-1]
        )

        self.mlp_layers = MLPLayers(size_list, dropout=self.dropout_prob, bn=True)
        self.predict_layer = nn.Linear(in_feature_num, 1)
        self.reg_loss = RegLoss()
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

    def cross_network(self, x_0):
        r"""Cross network is composed of cross layers, with each layer having the following formula.

        .. math:: x_{l+1} = x_0 {x_l^T} w_l + b_l + x_l

        :math:`x_l`, :math:`x_{l+1}` are column vectors denoting the outputs from the l -th and
        (l + 1)-th cross layers, respectively.
        :math:`w_l`, :math:`b_l` are the weight and bias parameters of the l -th layer.

        Args:
            x_0(torch.Tensor): Embedding vectors of all features, input of cross network.

        Returns:
            torch.Tensor:output of cross network, [batch_size, num_feature_field * embedding_size]

        """
        x_l = x_0
        for i in range(self.cross_layer_num):
            xl_w = torch.tensordot(x_l, self.cross_layer_w[i], dims=([1], [0]))
            xl_dot = (x_0.transpose(0, 1) * xl_w).transpose(0, 1)
            x_l = xl_dot + self.cross_layer_b[i] + x_l
        return x_l

    def forward(self, interaction):
        dcn_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]
        batch_size = dcn_all_embeddings.shape[0]
        dcn_all_embeddings = dcn_all_embeddings.view(batch_size, -1)

        # DNN
        deep_output = self.mlp_layers(dcn_all_embeddings)
        # Cross Network
        cross_output = self.cross_network(dcn_all_embeddings)
        stack = torch.cat([cross_output, deep_output], dim=-1)
        output = self.predict_layer(stack)

        return output.squeeze(1), dcn_all_embeddings, deep_output, cross_output

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        ratings = interaction['rating']
        output, dcn_all_embs, deep_output, cross_output = self.forward(interaction)
        l2_loss = self.reg_weight * self.reg_loss(self.cross_layer_w)
        if self.jaclip:
            lip_mat_deep = []
            # deep network
            for i in range(deep_output.shape[1]):
                v = torch.zeros_like(deep_output)
                v[:, i] = 1
                gradients = autograd.grad(outputs=deep_output, inputs=dcn_all_embs, grad_outputs=v,
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
                grad_norm = torch.norm(gradients, dim=1).unsqueeze(dim=1)
                lip_mat_deep.append(grad_norm)
            # # cross network
            # for i in range(cross_output.shape[1]):
            #     v = torch.zeros_like(cross_output)
            #     v[:, i] = 1
            #     gradients = autograd.grad(outputs=cross_output, inputs=dcn_all_embs, grad_outputs=v,
            #                             create_graph=True, retain_graph=True, only_inputs=True)[0]
            #     grad_norm = torch.norm(gradients, dim=1).unsqueeze(dim=1)
            #     lip_mat_cross.append(grad_norm)

            lip_concat_deep = torch.cat(lip_mat_deep, dim=1)
            lip_con_norm_deep = torch.norm(lip_concat_deep, dim=1)
            lip_loss_deep = torch.max(lip_con_norm_deep)
            # lip_concat_cross = torch.cat(lip_mat_cross, dim=1)
            # lip_con_norm_cross = torch.norm(lip_concat_cross, dim=1)
            # lip_loss_cross = torch.max(lip_con_norm_cross)
            # lip_loss = (lip_loss_deep + lip_loss_cross) / 2
            lip_loss = lip_loss_deep
            if not self.loss_type or self.loss_type == 'bce':
                return self.loss(output, label) + l2_loss, self.jaclip_weight * lip_loss
            elif self.loss_type == 'rating':
                return self.loss(output, ratings) + l2_loss, self.jaclip_weight * lip_loss
        
        if not self.loss_type or self.loss_type == 'bce':
            return self.loss(output, label) + l2_loss
        elif self.loss_type == 'rating':
            return self.loss(output, ratings) + l2_loss

    def predict(self, interaction):
        output, _, _, _ = self.forward(interaction)
        return self.sigmoid(output)
