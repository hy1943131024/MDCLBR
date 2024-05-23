#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
from random import random, sample
from utility import Datasets
# from utils import timer
def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs)) # [bs]
    loss = torch.mean(loss)

    return loss


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


class MDCLBR(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]
        self.status = conf["status"]
        self.init_emb()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph

        # generate the graph without any dropouts for testing
        self.get_item_level_graph_ori()
        self.get_bundle_level_graph_ori()
        self.get_bundle_agg_graph_ori()

        # generate the graph with the configured dropouts for training, if aug_type is OP or MD, the following graphs with be identical with the aboves
        self.get_item_level_graph()
        self.get_bundle_level_graph()
        self.get_bundle_agg_graph()

        self.init_md_dropouts()

        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]
        self.c_temp1 = self.conf["c_temps1"]
        self.ub_p_drop = self.conf["ub_p_drop"]
        self.mix_ratio = self.conf["mix_ratio"]
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_size , self.embedding_size , bias=True),
            nn.LeakyReLU(0.5),
        )
        self.mlp_r = nn.Sequential(
            nn.Linear(self.embedding_size , self.embedding_size , bias=True),
            nn.LeakyReLU(0.5),
        )


    def init_md_dropouts(self):
        self.item_level_dropout = nn.Dropout(self.conf["item_level_ratio"], True)
        self.bundle_level_dropout = nn.Dropout(self.conf["bundle_level_ratio"], True)
        self.bundle_agg_dropout = nn.Dropout(self.conf["bundle_agg_ratio"], True)


    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.users_feature_l0=self.users_feature
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.bundles_feature_l0=self.bundles_feature
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)


    def get_item_level_graph(self):
        ui_graph = self.ui_graph
        device = self.device
        modification_ratio = self.conf["item_level_ratio"]

        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = item_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.item_level_graph = to_tensor(laplace_transform(item_level_graph)).to(device)


    def get_item_level_graph_ori(self):
        ui_graph = self.ui_graph
        device = self.device
        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        self.item_level_graph_ori = to_tensor(laplace_transform(item_level_graph)).to(device)


    def get_bundle_level_graph(self):
        ub_graph = self.ub_graph
        device = self.device
        modification_ratio = self.conf["bundle_level_ratio"]

        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bundle_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bundle_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.bundle_level_graph = to_tensor(laplace_transform(bundle_level_graph)).to(device)


    def get_bundle_level_graph_ori(self):
        ub_graph = self.ub_graph
        device = self.device
        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
        self.bundle_level_graph_ori = to_tensor(laplace_transform(bundle_level_graph)).to(device)


    def get_bundle_agg_graph(self):
        bi_graph = self.bi_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = self.bi_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            bi_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph = to_tensor(bi_graph).to(device)


    def get_bundle_agg_graph_ori(self):
        bi_graph = self.bi_graph
        device = self.device

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device)


    def one_propagate(self, graph, A_feature, B_feature, mess_dropout, test):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test: # !!! important
                features = mess_dropout(features)

            features = features / (i+2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature


    def get_IL_bundle_rep(self, IL_items_feature, test):
        if test:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph_ori, IL_items_feature)
        else:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph, IL_items_feature)

        # simple embedding dropout on bundle embeddings
        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            IL_bundles_feature = self.bundle_agg_dropout(IL_bundles_feature)

        return IL_bundles_feature


    def propagate(self, test=False):
        #  =============================  item level propagation  =============================
        if test:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph_ori, self.users_feature, self.items_feature, self.item_level_dropout, test)
        else:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph, self.users_feature, self.items_feature, self.item_level_dropout, test)

        # aggregate the items embeddings within one bundle to obtain the bundle representation
        IL_bundles_feature = self.get_IL_bundle_rep(IL_items_feature, test)

        #  ============================= bundle level propagation =============================
        if test:
            BL_users_feature, BL_bundles_feature = self.one_propagate(self.bundle_level_graph_ori, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test)
        else:
            BL_users_feature, BL_bundles_feature = self.one_propagate(self.bundle_level_graph, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test)

        users_feature = [IL_users_feature, BL_users_feature]
        bundles_feature = [IL_bundles_feature, BL_bundles_feature]

        return users_feature, bundles_feature


    def cal_c_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1) # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score-pos_score))

        return c_loss

    def cal_loss(self, users_feature, bundles_feature,users_feature_r,bundles_feature_r):
        # IL: item_level, BL: bundle_level
        # [bs, 1, emb_size]
        IL_users_feature, BL_users_feature = users_feature
        # [bs, 1+neg_num, emb_size]
        IL_bundles_feature, BL_bundles_feature = bundles_feature
        # [bs, 1+neg_num]
        # users_feature_r = self.sim_evaluate(users_feature_r ,IL_users_feature, BL_users_feature)
        # bundles_feature_r = self.sim_evaluate_b(bundles_feature_r, IL_bundles_feature,
        #                                         BL_bundles_feature)
        # pred = torch.mm(users_feature_r, bundles_feature_r.t())
        pred = torch.sum(IL_users_feature * IL_bundles_feature, 2) + torch.sum(BL_users_feature * BL_bundles_feature, 2)
        bpr_loss = cal_bpr_loss(pred)
        # pred = torch.sum(torch.mm(self.users_feature_rr ,self.bundles_feature_rr.t()))
        # bpr_loss = cal_bpr_loss(pred)
        return bpr_loss
    def cal_loss_r(self, users_feature, bundles_feature):
        pred = torch.sum(users_feature[0] * bundles_feature[0], 2)
        bpr_loss_r = cal_bpr_loss(pred)

        return bpr_loss_r
    def get_bundle_level_graph1(self):
        ub_graph = self.ub_graph
        device = self.device
        modification_ratio = self.conf["bundle_level_ratio"]

        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bundle_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bundle_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.bundle_level_graph = to_tensor(laplace_transform(bundle_level_graph)).to(device)
    #对比学习loss定义
    # def info_nce_loss_overall(self, z1, z2, z_all):
    #
    #     self.tau=0.25
    #     f = lambda x: torch.exp(x / 0.25)
    #     f1 = lambda x: x
    #     # batch_size
    #     BL_ebedding_ub = z1[1][:,0,:]
    #     BL_ebedding_ui = z1[0][:, 0, :]
    #     # BL_ebedding = z1.squeeze(1)
    #     # [bs, 1+neg_num, emb_size]
    #     BL_all_ebedding = z_all
    #
    #     z2 = z2[0][:,0,:]
    #     # BL_ebedding_ub = BL_ebedding_ub + self.mlp(BL_ebedding_ub)
    #     # BL_ebedding_ui = BL_ebedding_ui + self.mlp_r(BL_ebedding_ui)
    #     # z2 = z2 + self.mlp(z2)
    #     # BL_all_ebedding = BL_all_ebedding + self.mlp(BL_all_ebedding)
    #     # BL_ebedding_ub = self.mlp(BL_ebedding_ub)
    #     # BL_ebedding_ui = self.mlp_r(BL_ebedding_ui)
    #     # BL_all_ebedding = self.mlp(BL_all_ebedding)
    #     # z2 = self.mlp(z2)
    #     # BL_ebedding = self.mlp(BL_ebedding)
    #     # z2 = self.mlp_r(z2)
    #     # BL_all_ebedding = self.mlp(BL_all_ebedding)
    #     # z2_all = self.mlp_r(z2_all)
    #     between_sim_ub = f1(self.sim(BL_ebedding_ub, z2))
    #     between_sim_ui = f1(self.sim(BL_ebedding_ui, z2))
    #     between_sim_ub_r=between_sim_ub/(between_sim_ub+between_sim_ui)
    #     between_sim_ui_r =between_sim_ui/(between_sim_ub+between_sim_ui)
    #     between_sim_ub_r = between_sim_ub_r.reshape(z1[0].size(0), 1)
    #     between_sim_ui_r = between_sim_ui_r.reshape(z1[0].size(0), 1)
    #     BL_ebedding =BL_ebedding_ub * between_sim_ub_r+BL_ebedding_ui * between_sim_ui_r
    #     # BL_ebedding = BL_ebedding+self.mlp(BL_ebedding)
    #     # z2 = z2+self.mlp(z2)
    #     # BL_all_ebedding = BL_all_ebedding+self.mlp(BL_all_ebedding)
    #     between_sim = f(self.sim(BL_ebedding, z2))
    #     # sim(batch_size, emb_dim || all_item, emb_dim) -> batch_size, all_item
    #     all_sim = f(self.sim(BL_ebedding, BL_all_ebedding))
    #     # all_sim_1 = f(self.sim(BL_ebedding, z2_all))
    #     # batch_size
    #     positive_pairs = between_sim
    #     # batch_size
    #     negative_pairs = torch.sum(all_sim, 1)
    #     # negative_pairs_r = torch.sum(all_sim_1, 1)
    #     loss = torch.sum(-torch.log(positive_pairs/(negative_pairs+1e-8)))
    #     return loss

    def info_nce_loss_overall(self, z1, z2, z_all1,z_all2):

        # self.tau = 0.25
        f = lambda x: torch.exp(x /  self.c_temp )
        f1 = lambda x: ((torch.exp(x/self.c_temp1)))
        # batch_size
        BL_ebedding_ub = z1[1][:, 0, :]
        BL_ebedding_ui = z1[0][:, 0, :]
        # BL_ebedding = z1.squeeze(1)
        # [bs, 1+neg_num, emb_size]
        BL_all_ebedding = z_all1
        re_all_ebedding = z_all2
        z2 = z2[0][:, 0, :]
        # BL_ebedding_ub = BL_ebedding_ub + self.mlp(BL_ebedding_ub)
        # BL_ebedding_ui = BL_ebedding_ui + self.mlp_r(BL_ebedding_ui)
        # z2 = z2 + self.mlp(z2)
        # BL_all_ebedding = BL_all_ebedding + self.mlp(BL_all_ebedding)
        # BL_ebedding_ub = self.mlp(BL_ebedding_ub)
        # BL_ebedding_ui = self.mlp_r(BL_ebedding_ui)
        # BL_all_ebedding = self.mlp(BL_all_ebedding)
        # z2 = self.mlp(z2)
        # BL_ebedding = self.mlp(BL_ebedding)
        # z2 = self.mlp_r(z2)
        # BL_all_ebedding = self.mlp(BL_all_ebedding)
        # z2_all = self.mlp_r(z2_all)
        between_sim_ub = f1(self.sim(BL_ebedding_ub, z2))
        between_sim_ui = f1(self.sim(BL_ebedding_ui, z2))
        between_sim_ub_r = between_sim_ub / (between_sim_ub + between_sim_ui)
        between_sim_ui_r = between_sim_ui / (between_sim_ub + between_sim_ui)
        between_sim_ub_r = between_sim_ub_r.reshape(z1[0].size(0), 1)
        between_sim_ui_r = between_sim_ui_r.reshape(z1[0].size(0), 1)
        BL_ebedding = BL_ebedding_ub * between_sim_ub_r + BL_ebedding_ui * between_sim_ui_r
        # BL_ebedding = BL_ebedding+self.mlp(BL_ebedding)
        # z2 = z2+self.mlp(z2)
        # BL_all_ebedding = BL_all_ebedding+self.mlp(BL_all_ebedding)
        between_sim = f(self.sim(BL_ebedding, z2))
        # sim(batch_size, emb_dim || all_item, emb_dim) -> batch_size, all_item
        all_sim1 = f(self.sim(BL_ebedding, BL_all_ebedding))
        all_sim2 = f(self.sim(BL_ebedding, re_all_ebedding))
        # all_sim_1 = f(self.sim(BL_ebedding, z2_all))
        # batch_size
        positive_pairs = between_sim
        # batch_size
        negative_pairs1 = torch.sum(all_sim1, 1)
        negative_pairs2 = torch.sum(all_sim2, 1)
        # negative_pairs_r = torch.sum(all_sim_1, 1)
        loss = torch.sum(-torch.log(positive_pairs / (negative_pairs1 + negative_pairs2-positive_pairs)))
        return loss
#   def info_nce_loss_overall_t(self, z1, z2, z_all):
#         # self.tau = 0.25
#         f = lambda x: torch.exp(x / self.c_temp )
#         # batch_size
#         # BL_ebedding_ub = z1[1][:, 0, :]
#         # BL_ebedding_ui = z1[0][:, 0, :]
#         # # BL_ebedding = z1.squeeze(1)
#         # # [bs, 1+neg_num, emb_size]
#         # BL_all_ebedding = z_all
#         z1 = z1[0][:, 0, :]
#         z2 = z2[0][:, 0, :]
#         # BL_ebedding_ub = BL_ebedding_ub+self.mlp(BL_ebedding_ub)
#         # BL_ebedding_ui = BL_ebedding_ui+self.mlp(BL_ebedding_ui)
#         # BL_all_ebedding = BL_all_ebedding+self.mlp(BL_all_ebedding)
#         # z2 = z2+self.mlp_r(z2)
#         # BL_ebedding = self.mlp(BL_ebedding)
#         # z2 = self.mlp_r(z2)
#         # BL_all_ebedding = self.mlp(BL_all_ebedding)
#         # z2_all = self.mlp_r(z2_all)
#         # between_sim_ub = f(self.sim(BL_ebedding_ub, z2))
#         # between_sim_ui = f(self.sim(BL_ebedding_ui, z2))
#         # between_sim_ub_r = between_sim_ub / (between_sim_ub + between_sim_ui)
#         # between_sim_ui_r = between_sim_ui / (between_sim_ub + between_sim_ui)
#         # between_sim_ub_r = between_sim_ub_r.reshape(2048, 1)
#         # between_sim_ui_r = between_sim_ui_r.reshape(2048, 1)
#         # BL_ebedding = BL_ebedding_ub * between_sim_ub_r + BL_ebedding_ui * between_sim_ui_r
#         between_sim = f(self.sim(z1, z2))
#         # sim(batch_size, emb_dim || all_item, emb_dim) -> batch_size, all_item
#         all_sim = f(self.sim(z1, z_all))
#         # all_sim_1 = f(self.sim(BL_ebedding, z2_all))
#         # batch_size
#         positive_pairs = between_sim
#         # batch_size
#         negative_pairs = torch.sum(all_sim, 1)
#         # negative_pairs1 = torch.sum(all_sim1, 1)
#         # negative_pairs_r = torch.sum(all_sim_1, 1)
#         loss = torch.sum(-torch.log(positive_pairs / (negative_pairs +1e-8)))
#         return loss
    def info_nce_loss_overall_t(self, z1, z2, z_all):
        # self.tau = 0.25
        f = lambda x: torch.exp(x / self.c_temp )
        # batch_size
        # BL_ebedding_ub = z1[1][:, 0, :]
        # BL_ebedding_ui = z1[0][:, 0, :]
        # # BL_ebedding = z1.squeeze(1)
        # # [bs, 1+neg_num, emb_size]
        # BL_all_ebedding = z_all
        z1 = z1[0][:, 0, :]
        z2 = z2[0][:, 0, :]
        # BL_ebedding_ub = BL_ebedding_ub+self.mlp(BL_ebedding_ub)
        # BL_ebedding_ui = BL_ebedding_ui+self.mlp(BL_ebedding_ui)
        # BL_all_ebedding = BL_all_ebedding+self.mlp(BL_all_ebedding)
        # z2 = z2+self.mlp_r(z2)
        # BL_ebedding = self.mlp(BL_ebedding)
        # z2 = self.mlp_r(z2)
        # BL_all_ebedding = self.mlp(BL_all_ebedding)
        # z2_all = self.mlp_r(z2_all)
        # between_sim_ub = f(self.sim(BL_ebedding_ub, z2))
        # between_sim_ui = f(self.sim(BL_ebedding_ui, z2))
        # between_sim_ub_r = between_sim_ub / (between_sim_ub + between_sim_ui)
        # between_sim_ui_r = between_sim_ui / (between_sim_ub + between_sim_ui)
        # between_sim_ub_r = between_sim_ub_r.reshape(2048, 1)
        # between_sim_ui_r = between_sim_ui_r.reshape(2048, 1)
        # BL_ebedding = BL_ebedding_ub * between_sim_ub_r + BL_ebedding_ui * between_sim_ui_r
        between_sim = f(self.sim(z1, z2))
        # sim(batch_size, emb_dim || all_item, emb_dim) -> batch_size, all_item
        all_sim = f(self.sim(z1, z_all))
        # all_sim1 = f(self.sim(z1, z_all1))
        # all_sim_1 = f(self.sim(BL_ebedding, z2_all))
        # batch_size
        positive_pairs = between_sim
        # batch_size
        negative_pairs = torch.sum(all_sim, 1)
        # negative_pairs1 = torch.sum(all_sim1, 1)
        # negative_pairs_r = torch.sum(all_sim_1, 1)
        loss = torch.sum(-torch.log(positive_pairs / (negative_pairs +1e-8)))
        return loss
    # def info_nce_loss_overall_t(self, z1, z2, z_all,z_all1):
    #
    #     # self.tau = 0.25
    #     f = lambda x: torch.exp(x / self.c_temp )
    #     # batch_size
    #     # BL_ebedding_ub = z1[1][:, 0, :]
    #     # BL_ebedding_ui = z1[0][:, 0, :]
    #     # # BL_ebedding = z1.squeeze(1)
    #     # # [bs, 1+neg_num, emb_size]
    #     # BL_all_ebedding = z_all
    #     z1 = z1[0][:, 0, :]
    #     z2 = z2[0][:, 0, :]
    #     # BL_ebedding_ub = BL_ebedding_ub+self.mlp(BL_ebedding_ub)
    #     # BL_ebedding_ui = BL_ebedding_ui+self.mlp(BL_ebedding_ui)
    #     # BL_all_ebedding = BL_all_ebedding+self.mlp(BL_all_ebedding)
    #     # z2 = z2+self.mlp_r(z2)
    #     # BL_ebedding = self.mlp(BL_ebedding)
    #     # z2 = self.mlp_r(z2)
    #     # BL_all_ebedding = self.mlp(BL_all_ebedding)
    #     # z2_all = self.mlp_r(z2_all)
    #     # between_sim_ub = f(self.sim(BL_ebedding_ub, z2))
    #     # between_sim_ui = f(self.sim(BL_ebedding_ui, z2))
    #     # between_sim_ub_r = between_sim_ub / (between_sim_ub + between_sim_ui)
    #     # between_sim_ui_r = between_sim_ui / (between_sim_ub + between_sim_ui)
    #     # between_sim_ub_r = between_sim_ub_r.reshape(2048, 1)
    #     # between_sim_ui_r = between_sim_ui_r.reshape(2048, 1)
    #     # BL_ebedding = BL_ebedding_ub * between_sim_ub_r + BL_ebedding_ui * between_sim_ui_r
    #     between_sim = f(self.sim(z1, z2))
    #     # sim(batch_size, emb_dim || all_item, emb_dim) -> batch_size, all_item
    #     all_sim = f(self.sim(z1, z_all))
    #     all_sim1 = f(self.sim(z1, z_all1))
    #     # all_sim_1 = f(self.sim(BL_ebedding, z2_all))
    #     # batch_size
    #     positive_pairs = between_sim
    #     # batch_size
    #     negative_pairs = torch.sum(all_sim, 1)
    #     negative_pairs1 = torch.sum(all_sim1, 1)
    #     # negative_pairs_r = torch.sum(all_sim_1, 1)
    #     loss = torch.sum(-torch.log(positive_pairs / (negative_pairs +negative_pairs1-positive_pairs)))
    #     return loss
    def info_nce_loss_overall_o(self, z1, z2, z_all):

        # self.tau = 0.25
        f = lambda x: torch.exp(x / self.c_temp )
        # batch_size
        # BL_ebedding_ub = z1[1][:, 0, :]
        # BL_ebedding_ui = z1[0][:, 0, :]
        # # BL_ebedding = z1.squeeze(1)
        # # [bs, 1+neg_num, emb_size]
        # BL_all_ebedding = z_all
        z1 = z1[:, 0, :]
        z2 = z2[:, 0, :]
        # BL_ebedding_ub = BL_ebedding_ub+self.mlp(BL_ebedding_ub)
        # BL_ebedding_ui = BL_ebedding_ui+self.mlp(BL_ebedding_ui)
        # BL_all_ebedding = BL_all_ebedding+self.mlp(BL_all_ebedding)
        # z2 = z2+self.mlp_r(z2)
        # BL_ebedding = self.mlp(BL_ebedding)
        # z2 = self.mlp_r(z2)
        # BL_all_ebedding = self.mlp(BL_all_ebedding)
        # z2_all = self.mlp_r(z2_all)
        # between_sim_ub = f(self.sim(BL_ebedding_ub, z2))
        # between_sim_ui = f(self.sim(BL_ebedding_ui, z2))
        # between_sim_ub_r = between_sim_ub / (between_sim_ub + between_sim_ui)
        # between_sim_ui_r = between_sim_ui / (between_sim_ub + between_sim_ui)
        # between_sim_ub_r = between_sim_ub_r.reshape(2048, 1)
        # between_sim_ui_r = between_sim_ui_r.reshape(2048, 1)
        # BL_ebedding = BL_ebedding_ub * between_sim_ub_r + BL_ebedding_ui * between_sim_ui_r
        between_sim = f(self.sim(z1, z2))
        # sim(batch_size, emb_dim || all_item, emb_dim) -> batch_size, all_item
        all_sim = f(self.sim(z1, z_all))
        # all_sim1 = f(self.sim(z1, z_all1))
        # all_sim_1 = f(self.sim(BL_ebedding, z2_all))
        # batch_size
        positive_pairs = between_sim
        # batch_size
        negative_pairs = torch.sum(all_sim, 1)
        # negative_pairs1 = torch.sum(all_sim1, 1)
        # negative_pairs_r = torch.sum(all_sim_1, 1)
        loss = torch.sum(-torch.log(positive_pairs / (negative_pairs +1e-8)))
        return loss
    def info_nce_loss_overall_oo(self, z1, z2, z_all):

        # self.tau = 0.25
        f = lambda x: torch.exp(x / self.c_temp )
        # batch_size
        # BL_ebedding_ub = z1[1][:, 0, :]
        # BL_ebedding_ui = z1[0][:, 0, :]
        # # BL_ebedding = z1.squeeze(1)
        # # [bs, 1+neg_num, emb_size]
        # BL_all_ebedding = z_all
        z1 = z1[:, 0, :]
        z2 = z2[0][:, 0, :]
        # BL_ebedding_ub = BL_ebedding_ub+self.mlp(BL_ebedding_ub)
        # BL_ebedding_ui = BL_ebedding_ui+self.mlp(BL_ebedding_ui)
        # BL_all_ebedding = BL_all_ebedding+self.mlp(BL_all_ebedding)
        # z2 = z2+self.mlp_r(z2)
        # BL_ebedding = self.mlp(BL_ebedding)
        # z2 = self.mlp_r(z2)
        # BL_all_ebedding = self.mlp(BL_all_ebedding)
        # z2_all = self.mlp_r(z2_all)
        # between_sim_ub = f(self.sim(BL_ebedding_ub, z2))
        # between_sim_ui = f(self.sim(BL_ebedding_ui, z2))
        # between_sim_ub_r = between_sim_ub / (between_sim_ub + between_sim_ui)
        # between_sim_ui_r = between_sim_ui / (between_sim_ub + between_sim_ui)
        # between_sim_ub_r = between_sim_ub_r.reshape(2048, 1)
        # between_sim_ui_r = between_sim_ui_r.reshape(2048, 1)
        # BL_ebedding = BL_ebedding_ub * between_sim_ub_r + BL_ebedding_ui * between_sim_ui_r
        between_sim = f(self.sim(z1, z2))
        # sim(batch_size, emb_dim || all_item, emb_dim) -> batch_size, all_item
        all_sim = f(self.sim(z1, z_all))
        # all_sim1 = f(self.sim(z1, z_all1))
        # all_sim_1 = f(self.sim(BL_ebedding, z2_all))
        # batch_size
        positive_pairs = between_sim
        # batch_size
        negative_pairs = torch.sum(all_sim, 1)
        # negative_pairs1 = torch.sum(all_sim1, 1)
        # negative_pairs_r = torch.sum(all_sim_1, 1)
        loss = torch.sum(-torch.log(positive_pairs / (negative_pairs+1e-8 )))
        return loss
    def embdding_r(self, z1, z2):
        # self.tau = 0.25
        f = lambda x: x
        between_sim_ub = f(self.sim(z1, z2[0]))
        between_sim_ui = f(self.sim(z1, z2[1]))
        between_sim_ub_r = between_sim_ub / (between_sim_ub + between_sim_ui)
        between_sim_ui_r = between_sim_ui / (between_sim_ub + between_sim_ui)
        between_sim_ub_r = between_sim_ub_r.reshape(z2[0].size(0), 1)
        between_sim_ui_r = between_sim_ui_r.reshape(z2[0].size(0), 1)
        z2_r = z2[0] * between_sim_ub_r + z2[1] * between_sim_ui_r
        return z2_r
    def sim_evaluate(self, z1, z2, z_all):

        # self.tau = 0.25
        f = lambda x: torch.exp(x /  self.c_temp )
        z1 = z1[0]
        between_sim_ub = f(self.sim(z2, z1))
        between_sim_ui = f(self.sim(z_all, z1))
        between_sim_ub_r = between_sim_ub / (between_sim_ub + between_sim_ui)
        between_sim_ui_r = between_sim_ui / (between_sim_ub + between_sim_ui)
        between_sim_ub_r = between_sim_ub_r.reshape(z1.size(0), 1)
        between_sim_ui_r = between_sim_ui_r.reshape(z1.size(0), 1)
        BL_ebedding = z2 * between_sim_ub_r + z_all * between_sim_ui_r
        BL_ebedding=(BL_ebedding+z1)/2.0
        return BL_ebedding

    def sim_evaluate_b(self, z1, z2, z_all):

        # self.tau = 0.25
        f = lambda x: torch.exp(x /  self.c_temp )
        between_sim_ub = f(self.sim(z2, z1))
        between_sim_ui = f(self.sim(z_all, z1))
        between_sim_ub_r = between_sim_ub / (between_sim_ub + between_sim_ui)
        between_sim_ui_r = between_sim_ui / (between_sim_ub + between_sim_ui)
        between_sim_ub_r = between_sim_ub_r.reshape(self.num_bundles, 1)
        between_sim_ui_r = between_sim_ui_r.reshape(self.num_bundles, 1)
        BL_ebedding = z2 * between_sim_ub_r + z_all * between_sim_ui_r
        BL_ebedding=(BL_ebedding+z1)/2.0
        return BL_ebedding
    #Tensor计算相似性c_loss
    # def info_nce_loss_overall_r(self, z1, z2, z_all):
    #
    #     self.tau=0.25
    #     f = lambda x: torch.exp(x / 0.25)
    #     f1 = lambda x: x
    #     # batch_size
    #     BL_ebedding = z1[0][:,0,:]
    #     # BL_ebedding = z1.squeeze(1)
    #     # [bs, 1+neg_num, emb_size]
    #
    #     # BL_all_ebedding_r = z2_all[1]
    #     z2_ui = z2[0][:, 0, :]
    #     z2_ub = z2[1][:,0,:]
    #     # BL_ebedding = self.mlp_r(BL_ebedding)
    #     # z2 = self.mlp(z2)
    #     # BL_all_ebedding = self.mlp_r(BL_all_ebedding)
    #     # BL_all_ebedding_r = self.mlp(BL_all_ebedding_r)
    #     between_sim_ui = f1(self.sim(BL_ebedding, z2_ui))
    #     between_sim_ub = f1(self.sim(BL_ebedding, z2_ub))
    #     between_sim_ub_r = between_sim_ub / (between_sim_ub + between_sim_ui)
    #     between_sim_ui_r = between_sim_ui / (between_sim_ub + between_sim_ui)
    #     between_sim_ub_r = between_sim_ub_r.reshape(z2[0].size(0), 1)
    #     between_sim_ui_r = between_sim_ui_r.reshape(z2[0].size(0), 1)
    #     z2 = z2_ub * between_sim_ub_r + z2_ui * between_sim_ui_r
    #     between_sim = f(self.sim(BL_ebedding, z2))
    #     # BL_all_ebedding_ub = z_all[1]
    #     # BL_all_ebedding_ui = z_all[0]
    #     # # sim(batch_size, emb_dim || all_item, emb_dim) -> batch_size, all_item
    #     # between_sim = f(self.sim(BL_ebedding, z2))
    #     # all_sim_ui = f(self.sim(BL_ebedding, BL_all_ebedding_ui))
    #     # all_sim_ub = f(self.sim(BL_ebedding, BL_all_ebedding_ub))
    #     # all_sim_ub_r = all_sim_ub / (all_sim_ub + all_sim_ui)
    #     # all_sim_ui_r = all_sim_ui / (all_sim_ub + all_sim_ui)
    #     # # all_sim_ub_r = all_sim_ub_r.reshape(2048, 1)
    #     # # all_sim_ui_r = all_sim_ui_r.reshape(2048, 1)
    #     # BL_all_ebedding = z2_ub * all_sim_ub_r + z2_ui * all_sim_ui_r
    #     # all_sim = f(self.sim(BL_ebedding, BL_all_ebedding))
    #     BL_all_ebedding = z_all
    #     all_sim = f(self.sim(BL_ebedding, BL_all_ebedding))
    #     # all_sim_r = f(self.sim(BL_ebedding, BL_all_ebedding_r))
    #     # batch_size
    #     positive_pairs = between_sim
    #     # batch_size
    #     negative_pairs = torch.sum(all_sim, 1)
    #     # negative_pairs_r = torch.sum(all_sim_r, 1)
    #     loss = torch.sum(-torch.log(positive_pairs / (negative_pairs+1e-8)))
    #     return loss
    def info_nce_loss_overall_r(self, z1, z2, z_all1,z_all2):

        # self.tau=0.25
        f = lambda x: torch.exp(x /  self.c_temp )
        f1 = lambda x: (torch.exp(x/self.c_temp1))
        # batch_size
        BL_ebedding = z1[0][:,0,:]
        # BL_ebedding = z1.squeeze(1)
        # [bs, 1+neg_num, emb_size]

        # BL_all_ebedding_r = z2_all[1]
        z2_ui = z2[0][:, 0, :]
        z2_ub = z2[1][:,0,:]
        # BL_ebedding = self.mlp_r(BL_ebedding)
        # z2 = self.mlp(z2)
        # BL_all_ebedding = self.mlp_r(BL_all_ebedding)
        # BL_all_ebedding_r = self.mlp(BL_all_ebedding_r)
        between_sim_ui = f1(self.sim(BL_ebedding, z2_ui))
        between_sim_ub = f1(self.sim(BL_ebedding, z2_ub))
        between_sim_ub_r = between_sim_ub / (between_sim_ub + between_sim_ui)
        between_sim_ui_r = between_sim_ui / (between_sim_ub + between_sim_ui)
        between_sim_ub_r = between_sim_ub_r.reshape(z2[0].size(0), 1)
        between_sim_ui_r = between_sim_ui_r.reshape(z2[0].size(0), 1)
        z2 = z2_ub * between_sim_ub_r + z2_ui * between_sim_ui_r
        between_sim = f(self.sim(BL_ebedding, z2))
        # BL_all_ebedding_ub = z_all[1]
        # BL_all_ebedding_ui = z_all[0]
        # # sim(batch_size, emb_dim || all_item, emb_dim) -> batch_size, all_item
        # between_sim = f(self.sim(BL_ebedding, z2))
        # all_sim_ui = f(self.sim(BL_ebedding, BL_all_ebedding_ui))
        # all_sim_ub = f(self.sim(BL_ebedding, BL_all_ebedding_ub))
        # all_sim_ub_r = all_sim_ub / (all_sim_ub + all_sim_ui)
        # all_sim_ui_r = all_sim_ui / (all_sim_ub + all_sim_ui)
        # # all_sim_ub_r = all_sim_ub_r.reshape(2048, 1)
        # # all_sim_ui_r = all_sim_ui_r.reshape(2048, 1)
        # BL_all_ebedding = z2_ub * all_sim_ub_r + z2_ui * all_sim_ui_r
        # all_sim = f(self.sim(BL_ebedding, BL_all_ebedding))
        BL_all_ebedding1 = z_all1
        all_sim1 = f(self.sim(BL_ebedding, BL_all_ebedding1))
        BL_all_ebedding2 = z_all2
        all_sim2 = f(self.sim(BL_ebedding, BL_all_ebedding2))
        # all_sim_r = f(self.sim(BL_ebedding, BL_all_ebedding_r))
        # batch_size
        positive_pairs = between_sim
        # batch_size
        negative_pairs1 = torch.sum(all_sim1, 1)
        negative_pairs2 = torch.sum(all_sim2, 1)
        # negative_pairs_r = torch.sum(all_sim_r, 1)
        loss = torch.sum(-torch.log(positive_pairs / (negative_pairs1+negative_pairs2-positive_pairs)))
        return loss
    #Tensor计算相似性c_loss
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        if z1.size()[0] == z2.size()[0]:
            return F.cosine_similarity(z1, z2)
        else:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())

    #借鉴
    #计算相似性决定要不要shanbian
    def get_ui_views_weighted(self, bundle_stabilities, stab_weight):
        # graph = self.gcn_model.Graph
        # n_users = self.gcn_model.num_users

        # kg probability of keep
        bundle_stabilities = torch.exp(bundle_stabilities)
        bundle_weights = (bundle_stabilities - bundle_stabilities.min()) / (bundle_stabilities.max() - bundle_stabilities.min())
        # bundle_weights1=sorted(bundle_weights,reverse=True)
        # print(bundle_weights1)
        # bundle_weights2=bundle_weights1[0:len(bundle_weights1)*30//100]
        # bundle_weights2= sorted(bundle_weights2, reverse=False)
        # print(bundle_weights2)
        bundle_weights = bundle_weights.where(bundle_weights > self.conf["sim_restruciton"], torch.ones_like(bundle_weights) * self.conf["sim_restruciton"])
        # bundle_weights=sorted(bundle_weights, reverse=False)
        # print(torch.mean(bundle_weights))
        # print(torch.mean(bundle_weights))
        # print(torch.min(bundle_weights))
        # print(torch.max(bundle_weights))
        # overall probability of keep
        # stab_weight4=torch.mean(stab_weight*bundle_weights)
        weights = (1-self.ub_p_drop)/torch.mean(stab_weight*bundle_weights)*(stab_weight*bundle_weights)
        # weights = weights.where(weights>0.3, torch.ones_like(weights) * 0.3)
        weights = weights.where(weights<self.conf["sim_weight"], torch.ones_like(weights) * self.conf["sim_weight"])
        # print(torch.mean(weights))
        # print(torch.min(weights))
        # print(torch.max(weights))
        bundle_mask = torch.bernoulli(weights).to(torch.bool)
        # print(f"keep ratio: {bundle_mask.sum()/bundle_mask.size()[0]:.2f}")
        # drop
        # print(f"keep ratio: {bundle_mask.sum() / bundle_mask.size()[0]:.2f}")
        g_weighted = self.ub_drop_weighted(bundle_mask)
        g_weighted.requires_grad = False
        return g_weighted

    def get_ui_views_weighted_u(self, bundle_stabilities, stab_weight):
        # graph = self.gcn_model.Graph
        # n_users = self.gcn_model.num_users

        # kg probability of keep
        bundle_stabilities = torch.exp(bundle_stabilities)
        bundle_weights = (bundle_stabilities - bundle_stabilities.min()) / (
                    bundle_stabilities.max() - bundle_stabilities.min())
        # bundle_weights1 = sorted(bundle_weights, reverse=True)
        # print(bundle_weights1)
        # bundle_weights2 = bundle_weights1[0:len(bundle_weights1) * 30 // 100]
        # bundle_weights2 = sorted(bundle_weights2, reverse=False)
        # print(bundle_weights2)
        bundle_weights = bundle_weights.where(bundle_weights > self.conf["sim_restruciton_u"],
                                              torch.ones_like(bundle_weights) * self.conf["sim_restruciton_u"])
        # bundle_weights=sorted(bundle_weights, reverse=False)
        # print(torch.mean(bundle_weights))
        # print(torch.mean(bundle_weights))
        # print(torch.min(bundle_weights))
        # print(torch.max(bundle_weights))
        # overall probability of keep
        # stab_weight4 = torch.mean(stab_weight * bundle_weights)
        # print(bundle_weights)
        # bundle_weights = bundle_weights.where(bundle_weights > self.conf["sim_restruciton"], torch.ones_like(bundle_weights) * self.conf["sim_restruciton"])
        # print(torch.mean(kg_weights))
        # print(torch.min(kg_weights))
        # print(torch.max(kg_weights))
        # overall probability of keep
        weights = (1 - self.ub_p_drop) / torch.mean(stab_weight * bundle_weights) * (stab_weight * bundle_weights)
        # weights = weights.where(weights>0.3, torch.ones_like(weights) * 0.3)
        weights = weights.where(weights < self.conf["sim_weight"], torch.ones_like(weights) * self.conf["sim_weight"])
        # print(torch.mean(weights))
        # print(torch.min(weights))
        # print(torch.max(weights))
        bundle_mask = torch.bernoulli(weights).to(torch.bool)
        # print(f"keep ratio: {bundle_mask.sum()/bundle_mask.size()[0]:.2f}")
        # drop
        # print(f"keep ratio: {bundle_mask.sum() / bundle_mask.size()[0]:.2f}")
        g_weighted = self.ub_drop_weighted_r(bundle_mask)
        g_weighted.requires_grad = False
        return g_weighted

    #调用上边方法嬗变，形成juzheng
    def ub_drop_weighted(self, bundle_mask):
        device = self.device
        num_users=self.conf["num_users"]
        num_items=self.conf["num_bundles"]
        # item_mask: [item_num]
        bundle_mask = bundle_mask.tolist()
        n_nodes = num_users + num_items
        # [interaction_num]
        item_np = self.conf["trainItem"]
        keep_idx = list()
        # overall sample rate = 0.4*0.9 = 0.36
        for i, j in enumerate(item_np.tolist()):
            if bundle_mask[j] and random() > 0.6:
            # if  random() >= 0.6:
                keep_idx.append(i)
            # add random samples
        interaction_random_sample = sample(list(range(len(item_np))), int(len(item_np) * self.mix_ratio))
        keep_idx = list(set(keep_idx + interaction_random_sample))
        # keep_idx = list(set(keep_idx))
        # print(f"finally keep ratio: {len(keep_idx) / len(item_np.tolist()):.2f}")
        keep_idx = np.array(keep_idx)
        # print(self.conf["trainUser"])
        user_np = self.conf["trainUser"][keep_idx]
        # print(f"finally keep ratio: {len(keep_idx) / len(item_np.tolist()):.2f}")
        item_np = item_np[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.conf["num_users"] )), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1)) + 1e-8
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        # to coo
        coo = adj_matrix.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        g = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).coalesce().to(device)
        g.requires_grad = False
        return g

    def ub_drop_weighted_r(self, bundle_mask):
        device = self.device
        num_users = self.conf["num_users"]
        num_items = self.conf["num_bundles"]
        # item_mask: [item_num]
        bundle_mask = bundle_mask.tolist()
        n_nodes = num_users + num_items
        # [interaction_num]
        user_np = self.conf["trainUser"]
        keep_idx = list()
        # overall sample rate = 0.4*0.9 = 0.36
        # for i, j in enumerate(item_np.tolist()):
        #     if j==22864: print("**********************************************************")

        for i, j in enumerate(user_np.tolist()):
            # if random() >= 0.6:
            if bundle_mask[j] and random() > 0.6:
                keep_idx.append(i)
            # add random samples
        interaction_random_sample = sample(list(range(len(user_np))), int(len(user_np) * self.mix_ratio))
        keep_idx = list(set(keep_idx + interaction_random_sample))
        # # interaction_random_sample = sample(list(range(len(user_np))), int(len(user_np) * self.mix_ratio))
        # keep_idx = list(set(keep_idx))
        # print(f"finally keep ratio: {len(keep_idx) / len(user_np.tolist()):.2f}")
        keep_idx = np.array(keep_idx)
        # print(self.conf["trainUser"])
        item_np = self.conf["trainItem"][keep_idx]
        user_np = user_np[keep_idx]
        ratings = np.ones_like(item_np, dtype=np.float32)
        # tmp_adj = sp.csr_matrix((ratings, (user_np+self.conf["num_bundles"], item_np)), shape=(n_nodes, n_nodes))
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.conf["num_users"])), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T
        # adj_mat = adj_mat.T
        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1)) + 1e-8
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        # to coo
        coo = adj_matrix.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        g = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).coalesce().to(device)
        g.requires_grad = False
        return g

    #返回qianru
    def view_computer_ui(self, g_droped):
        """
        propagate methods for contrastive lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(self.kg_dict)
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def forward(self, batch, ED_drop=False):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
        if ED_drop:
            self.get_item_level_graph()
            self.get_bundle_level_graph()
            self.get_bundle_agg_graph()

        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        
        users, bundles = batch
        users_feature, bundles_feature = self.propagate()
        uiv1,uiv2=self.restronction_graph(users_feature,bundles_feature)
        test = False
        if self.status == 1:
            test=True
        self.BL_users_feature_r, self.BL_bundles_feature_r = self.one_propagate(uiv1, self.users_feature, self.bundles_feature, self.bundle_level_dropout,test)
        self.BL_users_feature_r_u, self.BL_bundles_feature_r_u = self.one_propagate(uiv2, self.users_feature,self.bundles_feature,self.bundle_level_dropout, test)
        # self.users_feature_rr = self.embdding_r(self.BL_users_feature_r,users_feature)
        # self.bundles_feature_rr = self.embdding_r(self.BL_bundles_feature_r, bundles_feature)
        # self.users_feature_rr_u = self.embdding_r(self.BL_users_feature_r_u, users_feature)
        # self.bundles_feature_rr_u = self.embdding_r(self.BL_bundles_feature_r_u, bundles_feature)
        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feature]
        bundles_embedding = [i[bundles] for i in bundles_feature]
        self.bundles_embedding_r = [self.BL_bundles_feature_r[bundles]]
        self.users_embedding_r = [self.BL_users_feature_r[users]]
        self.bundles_embedding_r_u = [self.BL_bundles_feature_r_u[bundles]]
        self.users_embedding_r_u = [self.BL_users_feature_r_u[users]]
        bpr_loss = self.cal_loss(users_embedding, bundles_embedding,self.users_embedding_r,self.bundles_embedding_r)
        # ###########
        # IL_users_feature, BL_users_feature = users_feature
        # IL_bundles_feature, BL_bundles_feature = bundles_feature
        # # sim_bundle = self.sim(IL_bundles_feature, BL_bundles_feature)
        # # sim = self.sim(IL_users_feature, IL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_feature)
        # # uiv1 = self.get_ui_views_weighted(sim,1)
        # sim_user_u_r = self.sim(self.BL_users_feature_r_u, BL_users_feature)
        # sim_bundle_b_r = self.sim(self.BL_bundles_feature_r, BL_bundles_feature)
        # # print(type(sim_user))
        # print(len(sim_user_u_r))
        # print(len(sim_bundle_b_r))
        # # print(min(sim_user))
        # # print(max(sim_user))
        # bins = np.array([-1.0, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # # 计算直方图
        # hist_u_r, edges_u_r = np.histogram(sim_user_u_r.cpu().detach().numpy(), bins)
        # hist_b_r, edges_b_r = np.histogram(sim_bundle_b_r.cpu().detach().numpy(), bins)
        # # 输出结果
        # print(hist_u_r)
        # print(hist_b_r)
        # bpr_loss = self.cal_loss(self.users_feature_rr, self.bundles_feature_rr)
        # bpr_loss_r = self.cal_loss_r(self.users_embedding_r, self.bundles_embedding_r)
        # c_u_loss = self.info_nce_loss_overall(users_embedding,self.users_embedding_r ,self.users_feature_rr)
        # c_b_loss = self.info_nce_loss_overall(bundles_embedding ,self.bundles_embedding_r, self.bundles_feature_rr)
        # c_u_loss_u = self.info_nce_loss_overall(users_embedding, self.users_embedding_r_u,self.users_feature_rr)
        # c_b_loss_u = self.info_nce_loss_overall(bundles_embedding, self.bundles_embedding_r_u, self.bundles_feature_rr)
        # c_u_loss = self.info_nce_loss_overall(users_embedding, self.users_embedding_r,self.users_feature_rr,self.BL_users_feature_r)
        # c_b_loss = self.info_nce_loss_overall(bundles_embedding, self.bundles_embedding_r, self.bundles_feature_rr,self.BL_bundles_feature_r)
        # c_u_loss_u = self.info_nce_loss_overall(users_embedding, self.users_embedding_r_u, self.users_feature_rr_u,self.BL_users_feature_r_u)
        # c_b_loss_u = self.info_nce_loss_overall(bundles_embedding, self.bundles_embedding_r_u, self.bundles_feature_rr_u,self.BL_bundles_feature_r_u)
        # c_u_loss_u_r = self.info_nce_loss_overall_t(self.users_embedding_r, self.users_embedding_r_u,
        #                                             self.BL_users_feature_r)
        # c_b_loss_u_r = self.info_nce_loss_overall_t(self.bundles_embedding_r, self.bundles_embedding_r_u,
        #                                             self.BL_bundles_feature_r)
        # c_u_loss_u_rr = self.info_nce_loss_overall_t(self.users_embedding_r_u, self.users_embedding_r,
        #                                              self.BL_users_feature_r_u)
        # c_b_loss_u_rr = self.info_nce_loss_overall_t(self.bundles_embedding_r_u, self.bundles_embedding_r,
        #                                              self.BL_bundles_feature_r_u)
        c_u_loss_u_r = self.info_nce_loss_overall_t(self.users_embedding_r, self.users_embedding_r_u,
                                                    self.BL_users_feature_r)
        c_b_loss_u_r = self.info_nce_loss_overall_t(self.bundles_embedding_r, self.bundles_embedding_r_u,
                                                    self.BL_bundles_feature_r)
        c_u_loss_u_rr = self.info_nce_loss_overall_t(self.users_embedding_r_u, self.users_embedding_r,
                                                     self.BL_users_feature_r_u)
        c_b_loss_u_rr = self.info_nce_loss_overall_t(self.bundles_embedding_r_u, self.bundles_embedding_r,
                                                     self.BL_bundles_feature_r_u)
        # c_u_loss_u_r = self.info_nce_loss_overall_t(self.users_embedding_r, self.users_embedding_r_u, self.BL_users_feature_r, self.BL_users_feature_r_u)
        # c_b_loss_u_r = self.info_nce_loss_overall_t(self.bundles_embedding_r, self.bundles_embedding_r_u,self.BL_bundles_feature_r,  self.BL_bundles_feature_r_u)
        # c_u_loss_u_rr = self.info_nce_loss_overall_t(self.users_embedding_r_u,self.users_embedding_r, self.BL_users_feature_r_u, self.BL_users_feature_r)
        # c_b_loss_u_rr = self.info_nce_loss_overall_t(self.bundles_embedding_r_u,self.bundles_embedding_r,  self.BL_bundles_feature_r_u,self.BL_bundles_feature_r)
        # c_u_loss_u_o = self.info_nce_loss_overall_o(users_embedding[0], users_embedding[1],
        #                                             users_feature[0])
        # c_u_loss_u_oo = self.info_nce_loss_overall_o(users_embedding[1], users_embedding[0],
        #                                             users_feature[1])
        # c_b_loss_u_o = self.info_nce_loss_overall_o(bundles_embedding[0], bundles_embedding[1],
        #                                              bundles_feature[0])
        # c_b_loss_u_oo = self.info_nce_loss_overall_o(bundles_embedding[1], bundles_embedding[0],
        #                                              bundles_feature[1])
        c_u_loss_u_o = self.info_nce_loss_overall_o(users_embedding[0], users_embedding[1],
                                                    users_feature[0])
        c_u_loss_u_oo = self.info_nce_loss_overall_o(users_embedding[1], users_embedding[0],
                                                     users_feature[1])
        c_b_loss_u_o = self.info_nce_loss_overall_o(bundles_embedding[0], bundles_embedding[1],
                                                    bundles_feature[0])
        c_b_loss_u_oo = self.info_nce_loss_overall_o(bundles_embedding[1], bundles_embedding[0],
                                                     bundles_feature[1])
        c_u_loss_u_o_r = self.info_nce_loss_overall_oo(users_embedding[1],self.users_embedding_r,
                                                    users_feature[1])
        c_b_loss_u_o_r = self.info_nce_loss_overall_oo(bundles_embedding[1], self.bundles_embedding_r,
                                                    bundles_feature[1])
        c_u_loss_u_oo_rr = self.info_nce_loss_overall_oo(users_embedding[1], self.users_embedding_r_u,
                                                     users_feature[1])
        c_b_loss_u_oo_rr = self.info_nce_loss_overall_oo(bundles_embedding[1], self.bundles_embedding_r_u,
                                                     bundles_feature[1])
        ############################################
        c_u_loss_u_o_r_i = self.info_nce_loss_overall_oo(users_embedding[0], self.users_embedding_r,
                                                       users_feature[0])
        c_b_loss_u_o_r_i = self.info_nce_loss_overall_oo(bundles_embedding[0], self.bundles_embedding_r,
                                                       bundles_feature[0])
        c_u_loss_u_oo_rr_i = self.info_nce_loss_overall_oo(users_embedding[0], self.users_embedding_r_u,
                                                         users_feature[0])
        c_b_loss_u_oo_rr_i = self.info_nce_loss_overall_oo(bundles_embedding[0], self.bundles_embedding_r_u,
                                                         bundles_feature[0])
        ############################################

        # c_u_loss_r = self.info_nce_loss_overall_r(self.users_embedding_r, users_embedding, self.BL_users_feature_r)
        # c_b_loss_r = self.info_nce_loss_overall_r(self.bundles_embedding_r, bundles_embedding, self.BL_bundles_feature_r)
        # c_u_loss_r_u = self.info_nce_loss_overall_r(self.users_embedding_r_u, users_embedding, self.BL_users_feature_r_u)
        # c_b_loss_r_u = self.info_nce_loss_overall_r(self.bundles_embedding_r_u, bundles_embedding, self.BL_bundles_feature_r_u)
        # c_u_loss_r = self.info_nce_loss_overall_r(self.users_embedding_r, users_embedding, self.users_feature_rr,self.BL_users_feature_r)
        # c_b_loss_r = self.info_nce_loss_overall_r(self.bundles_embedding_r, bundles_embedding,self.bundles_feature_rr ,self.BL_bundles_feature_r)
        # c_u_loss_r_u = self.info_nce_loss_overall_r(self.users_embedding_r_u, users_embedding, self.users_feature_rr_u,self.BL_users_feature_r_u)
        # c_b_loss_r_u = self.info_nce_loss_overall_r(self.bundles_embedding_r_u, bundles_embedding,self.bundles_feature_rr_u ,self.BL_bundles_feature_r_u)
        print("zheshicross")
        length_user=users_feature[0].size(0)
        length_bundle = bundles_feature[0].size(0)
        length_user_r = users_feature[0].size(0)
        length_bundle_r = bundles_feature[0].size(0)
        # c_u_loss=torch.sum(c_u_loss,axis=0)
        # c_b_loss=torch.sum(c_b_loss,axis=0)
        # c_loss=c_u_loss / length_user*0.5+c_b_loss / length_bundle*0.5
        # c_loss = c_b_loss / length_bundle
        # c_loss = c_u_loss / length_user
        # c_u_loss_r = torch.sum(c_u_loss_r, axis=0)
        # c_b_loss_r = torch.sum(c_b_loss_r, axis=0)
        # c_loss_r = c_u_loss_r/ length_user_r*0.5  +c_b_loss_r/length_bundle_r*0.5
        # c_loss_r = c_u_loss_r / length_user
        # c_u_loss_u = torch.sum(c_u_loss_u, axis=0)
        # c_b_loss_u = torch.sum(c_b_loss_u, axis=0)
        # c_loss_u = c_u_loss_u / length_user_r
        # c_loss_u = c_b_loss_u / length_bundle
        # c_loss_u = c_u_loss_u / length_user_r *0.5 + c_b_loss_u / length_bundle_r*0.5
        c_u_loss_u_r = torch.sum(c_u_loss_u_r, axis=0)
        c_b_loss_u_r = torch.sum(c_b_loss_u_r, axis=0)
        c_loss_u_r = c_u_loss_u_r / length_user_r + c_b_loss_u_r / length_bundle_r
        c_u_loss_u_rr = torch.sum(c_u_loss_u_rr, axis=0)
        c_b_loss_u_rr = torch.sum(c_b_loss_u_rr, axis=0)
        c_loss_u_rr = c_u_loss_u_rr / length_user_r  + c_b_loss_u_rr / length_bundle_r
        ###################################
        c_u_loss_u_o = torch.sum(c_u_loss_u_o, axis=0)
        c_b_loss_u_o = torch.sum(c_b_loss_u_o, axis=0)
        c_u_loss_u_o = c_u_loss_u_o / length_user_r + c_b_loss_u_o / length_bundle_r
        c_u_loss_u_oo = torch.sum(c_u_loss_u_oo, axis=0)
        c_b_loss_u_oo = torch.sum(c_b_loss_u_oo, axis=0)
        c_u_loss_u_oo = c_u_loss_u_oo / length_user_r  + c_b_loss_u_oo / length_bundle_r
        c_u_loss_u_o_r = torch.sum(c_u_loss_u_o_r, axis=0)
        c_b_loss_u_o_r = torch.sum(c_b_loss_u_o_r, axis=0)
        c_u_loss_u_o_r = c_u_loss_u_o_r / length_user_r  + c_b_loss_u_o_r / length_bundle_r
        c_u_loss_u_oo_rr = torch.sum(c_u_loss_u_oo_rr, axis=0)
        c_b_loss_u_oo_rr = torch.sum(c_b_loss_u_oo_rr, axis=0)
        c_u_loss_u_oo_rr = c_u_loss_u_oo_rr / length_user_r+ c_b_loss_u_oo_rr / length_bundle_r
        c_u_loss_u_o_r_i = torch.sum(c_u_loss_u_o_r_i, axis=0)
        c_b_loss_u_o_r_i = torch.sum(c_b_loss_u_o_r_i, axis=0)
        c_u_loss_u_o_r_i = c_u_loss_u_o_r_i / length_user_r + c_b_loss_u_o_r_i / length_bundle_r
        c_u_loss_u_oo_rr_i = torch.sum(c_u_loss_u_oo_rr_i, axis=0)
        c_b_loss_u_oo_rr_i = torch.sum(c_b_loss_u_oo_rr_i, axis=0)
        c_u_loss_u_oo_rr_i = c_u_loss_u_oo_rr_i / length_user_r + c_b_loss_u_oo_rr_i / length_bundle_r
        ###################################
        # c_loss_u_r = c_u_loss_u_r / length_user_r + c_b_loss_u_r / length_bundle_r
        # c_loss_u =  c_b_loss_u / length_bundle_r
        # c_loss_u_r = c_u_loss_u_r / length_user_r
        # c_u_loss_r_u = torch.sum(c_u_loss_r_u, axis=0)
        # c_b_loss_r_u = torch.sum(c_b_loss_r_u, axis=0)
        # c_loss_r_u = c_u_loss_r_u / length_user_r*0.5 +c_b_loss_r_u/length_bundle_r*0.5
        # c_loss_r_s=c_loss_r+c_loss_r_u
        # c_loss_s = c_loss + c_loss_u
        # c_loss_r_u = c_b_loss_r_u / length_bundle
        # self.users_embedding_r[0] = self.mlp_r(self.users_embedding_r[0])
        # self.bundles_embedding_r[0] = self.mlp_r(self.bundles_embedding_r[0])
        # users_embedding[0]= self.mlp(users_embedding[0])
        # users_embedding[1] = self.mlp(users_embedding[1])
        # bundles_embedding[0] = self.mlp(bundles_embedding[0])
        # bundles_embedding[1] = self.mlp(bundles_embedding[1])
        return bpr_loss, c_loss_u_r,c_loss_u_rr,c_u_loss_u_o,c_u_loss_u_oo,c_u_loss_u_o_r,c_u_loss_u_oo_rr,c_u_loss_u_o_r_i,c_u_loss_u_oo_rr_i
    def restronction_graph(self,users_feature,bundles_feature):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop

        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        # users_feature, bundles_feature = self.propagate()
        # print(type(users_feature))
        # user_tensor=len()
        # users, bundles=users_feature.size(0)-1,bundles_feature.size(0)-1,
        # users, bundles = users_feature.size(0) - 1, bundles_feature.size(0) - 1,
        # print(users_feature.length(0))
        # users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feature]
        # bundles_embedding = [i[bundles] for i in bundles_feature]
        IL_users_feature, BL_users_feature = users_feature
        IL_bundles_feature, BL_bundles_feature = bundles_feature
        sim_bundle = self.sim(IL_bundles_feature, BL_bundles_feature)
        # sim = self.sim(IL_users_feature, IL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_featureIL_bundles_feature)
        # uiv1 = self.get_ui_views_weighted(sim,1)
        sim_user = self.sim(IL_users_feature,BL_users_feature)
        # print(type(sim_user))
        # print(len(sim_user))
        # print(len(sim_bundle))
        # # print(min(sim_user))
        # # print(max(sim_user))
        # bins = np.array([-1.0, -0.5, -0.4,-0.3, -0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9, 1.0])
        # # 计算直方图
        # hist_u, edges_u = np.histogram(sim_user.cpu().detach().numpy(), bins)
        # hist_b, edges_b = np.histogram(sim_bundle.cpu().detach().numpy(), bins)
        # # 输出结果
        # print(hist_u)
        # print(hist_b)
        uiv1 = self.get_ui_views_weighted(sim_bundle, 1)
        uiv2 = self.get_ui_views_weighted_u(sim_user, 1)
        return uiv1,uiv2

    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature = propagate_result
        users_feature_atom, users_feature_non_atom = [i[users] for i in users_feature]
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature
        scores = torch.mm(users_feature_atom, bundles_feature_atom.t()) + torch.mm(users_feature_non_atom, bundles_feature_non_atom.t())
        # self.users_embedding_r = [self.BL_users_feature_r[users]]
        # users_feature_atom, users_feature_non_atom = [i[users] for i in users_feature]
        # users_feature_r= [i[users] for i in self.users_embedding_r]
        # BL_bundles_feature_r=self.BL_bundles_feature_r
        # users_feature_r=users_feature_r[0].squeeze()
        # scores_r=torch.mm(users_feature_r, BL_bundles_feature_r.t())
        # scores = torch.mm(self.users_feature_rr, self.users_feature_rr)
        return scores



