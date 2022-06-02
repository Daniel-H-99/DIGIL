import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GNN import GNN
from model.GNN2 import GNN2
from model.AttDGI import AttDGI
from model.myDGI import myDGI
class BiGI(nn.Module):
    def __init__(self, opt):
        super(BiGI, self).__init__()
        self.opt=opt
        self.GNN = GNN(opt) # fast mode(GNN), slow mode(GNN2)
        if self.opt["number_user"] * self.opt["number_item"] > 10000000:
            self.DGI = AttDGI(opt) # Since pytorch is not support sparse matrix well
        else :
            self.DGI = myDGI(opt) # Since pytorch is not support sparse matrix well
        self.dropout = opt["dropout"]

        self.user_embedding = nn.Embedding(opt["number_user"], opt["feature_dim"])
        self.item_embedding = nn.Embedding(opt["number_item"], opt["feature_dim"])
        self.item_index = torch.arange(0, self.opt["number_item"], 1)
        self.user_index = torch.arange(0, self.opt["number_user"], 1)
        if self.opt["cuda"]:
            self.item_index = self.item_index.cuda()
            self.user_index = self.user_index.cuda()

    def score_predict(self, fea):
        out = self.GNN.score_function1(fea)
        out = F.relu(out)
        out = self.GNN.score_function2(out)
        out = torch.sigmoid(out)
        return out.view(out.size()[0], -1)

    def score(self, fea):
        out = self.GNN.score_function1(fea)
        out = F.relu(out)
        out = self.GNN.score_function2(out)
        out = torch.sigmoid(out)
        return out.view(-1)

    # def extend_user(self, num=1):
        
    def forward(self, ufea, vfea, UV_adj, VU_adj, adj):
        learn_user,learn_item = self.GNN(ufea,vfea,UV_adj,VU_adj,adj)
        return learn_user,learn_item

class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.opt = opt
        self.dim = opt['hidden_dim']
        self.num_labels = opt['number_label']
        self.layers = nn.Sequential(
            nn.Linear(self.dim, self.num_labels)
        )
        if self.opt['cuda']:
            self.layers = self.layers.cuda()
    
    def forward(self, input):
        return self.layers(input)

class Completer(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.opt = opt
        self.dim = opt['hidden_dim']
        self.num_labels = opt['number_label']
        self.W_q = nn.Linear(self.dim, self.dim)
        self.W_k = nn.Linear(self.dim, self.dim)
        self.T = 8
        if self.opt['cuda']:
            self.W_q = self.W_q.cuda()
            self.W_k = self.W_k.cuda()
    
    def forward(self, user, item, ignore):
        Q = self.W_q(user)
        K = self.W_k(item)
        attn =  torch.einsum('up,ip->ui') / self.T
        attn[ignore] = -1e9
        return attn