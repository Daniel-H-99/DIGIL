import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import torch_utils
from model.BiGI import BiGI, Classifier, Completer

class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):  # here should change
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

class DGILTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.model = BiGI(opt)
        self.classifier = Classifier(opt)
        self.completer = Completer(opt)
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_comp = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])
        self.optimizer_cls = torch_utils.get_optimizer(opt['optim'], self.classifier.parameters(), opt['lr_cls'])
        self.optimizer_comp = torch_utils.get_optimizer(opt['optim'], self.completer.parameters(), opt['lr_comp'])

        self.epoch_rec_loss = []
        self.epoch_dgi_loss = []
        self.epoch_cls_loss = []
        self.epoch_comp_loss = []

    def unpack_batch_predict(self, batch, cuda):
        batch = batch[0]
        if cuda:
            user_index = batch.cuda()
        else:
            user_index = batch
        return user_index

    def unpack_batch(self, batch, cuda):
        if cuda:
            inputs = [Variable(b.cuda()) for b in batch]
            user_index = inputs[0]
            item_index = inputs[1]
            negative_item_index = inputs[2]
        else:
            inputs = [Variable(b) for b in batch]
            user_index = inputs[0]
            item_index = inputs[1]
            negative_item_index = inputs[2]
        return user_index, item_index, negative_item_index

    def unpack_batch_DGI(self, batch, cuda):
        if cuda:
            user_index = batch[0].cuda()
            item_index = batch[1].cuda()
            negative_item_index = batch[2].cuda()
            User_index_One = batch[3].cuda()
            Item_index_One = batch[4].cuda()
            real_user_index_id_Two = batch[5].cuda()
            fake_user_index_id_Two = batch[6].cuda()
            real_item_index_id_Two = batch[7].cuda()
            fake_item_index_id_Two = batch[8].cuda()
        else:
            user_index = batch[0]
            item_index = batch[1]
            negative_item_index = batch[2]
            User_index_One = batch[3]
            Item_index_One = batch[4]
            real_user_index_id_Two = batch[5]
            fake_user_index_id_Two = batch[6]
            real_item_index_id_Two = batch[7]
            fake_item_index_id_Two = batch[8]
        return user_index, item_index, negative_item_index, User_index_One, Item_index_One, real_user_index_id_Two, fake_user_index_id_Two, real_item_index_id_Two, fake_item_index_id_Two

    def predict(self, batch):
        User_One = self.unpack_batch_predict(batch, self.opt["cuda"])  # 1

        Item_feature = torch.index_select(self.item_hidden_out, 0, self.model.item_index) # item_num * hidden_dim
        User_feature = torch.index_select(self.user_hidden_out, 0, User_One) # User_num * hidden_dim
        User_feature = User_feature.unsqueeze(1)
        User_feature = User_feature.repeat(1, self.opt["number_item"], 1)
        Item_feature = Item_feature.unsqueeze(0)
        Item_feature = Item_feature.repeat(User_feature.size()[0], 1, 1)
        Feature = torch.cat((User_feature, Item_feature),
                            dim=-1)
        output = self.model.score_predict(Feature)
        output_list, recommendation_list = output.sort(descending=True)
        return recommendation_list.cpu().numpy()

    def feature_corruption(self):
        user_index = torch.randperm(self.opt["number_user"], device=self.model.user_index.device)
        item_index = torch.randperm(self.opt["number_item"], device=self.model.user_index.device)
        user_feature = self.model.user_embedding(user_index)
        item_feature = self.model.item_embedding(item_index)
        return user_feature, item_feature

    def update_bipartite(self, static_user_feature, static_item_feature, UV_adj, VU_adj, adj,fake = 0):
        # We do not use any side information. if have side information, modify following codes.
        if fake:
            user_feature, item_feature = self.feature_corruption()
            user_feature = user_feature.detach()
            item_feature = item_feature.detach()
        else :
            user_feature = self.model.user_embedding(self.model.user_index)
            item_feature = self.model.item_embedding(self.model.item_index)

        self.user_hidden_out, self.item_hidden_out = self.model(user_feature, item_feature, UV_adj, VU_adj, adj)

    def HingeLoss(self, pos, neg):
        gamma = torch.tensor(self.opt["margin"])
        if self.opt["cuda"]:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        # import pdb
        # pdb.set_trace()
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def classify(self, batch):
        self.model.train()
        self.classifier.train()
        self.optimizer.zero_grad()
        self.optimizer_cls.zero_grad()

        self.update_bipartite(user_feature,item_feature, UV, VU, adj)
        user_hidden_out = self.user_hidden_out
        pred = self.classifier(user_hidden_out[batch['food']])

        loss = self.opt['lambda_cls'] * self.criterion_cls(pred, batch['ans'])
        loss.backward()
        self.optimizer.step()
        self.optimizer_cls.step()
        self.epoch_cls_loss.append(loss.item())

        return loss.item()

        # self.update_bipartite(user_feature, item_feature, CUV, CVU, fake_adj, fake = 1)

    def classify_pred(self, batch):
        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            self.update_bipartite(user_feature,item_feature, UV, VU, adj)
            user_hidden_out = self.user_hidden_out
            pred = self.classifier(user_hidden_out[batch['food']])
            pred_labels = pred.argmax(dim=1)
        return pred_labels

    def complete(self, batch):
        self.model.train()
        self.completer.train()
        self.optimizer.zero_grad()
        self.optimizer_completer.zero_grad()

        self.update_bipartite(user_feature,item_feature, UV, VU, adj)
        user_hidden_out = self.user_hidden_out
        item_hidden_out = self.item_hidden_out
        user_embedding, item_embedding = user_hidden_out[batch['food']], item_hidden_out[self.model.item_index]
        pred = self.completer(user_embedding, item_embedding, batch['ignore']) # B x num_food
        loss = self.opt['lambda_comp'] * self.criterion_comp(pred, batch['ans'])
        
        loss.backward()
        self.optimizer.step()
        self.optimizer_comp.step()
        self.epoch_comp_loss.append(loss.item())

        return loss.item()

    def complete_pred(self, batch):
        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            self.update_bipartite(user_feature,item_feature, UV, VU, adj)
            user_hidden_out = self.user_hidden_out
            item_hidden_out = self.item_hidden_out
            user_embedding, item_embedding = user_hidden_out[batch['food']], item_hidden_out[self.model.item_index]
            pred = self.completer(user_embedding, item_embedding, batch['ignore']) # B x num_food
            pred_labels = pred.argmax(dim=1)
        return pred_labels

    def reconstruct(self, UV, VU, adj, CUV, CVU, fake_adj, user_feature, item_feature, batch): 
        self.model.train()
        self.optimizer.zero_grad()

        self.update_bipartite(user_feature, item_feature, CUV, CVU, fake_adj, fake = 1)
        fake_user_hidden_out = self.user_hidden_out
        fake_item_hidden_out = self.item_hidden_out

        self.update_bipartite(user_feature,item_feature, UV, VU, adj)
        user_hidden_out = self.user_hidden_out
        item_hidden_out = self.item_hidden_out

        if self.opt["number_user"] * self.opt["number_item"] > 10000000:
            user_One, item_One, neg_item_One, User_index_One, Item_index_One, real_user_index_id_Two, fake_user_index_id_Two, real_item_index_id_Two, fake_item_index_id_Two  = self.unpack_batch_DGI(batch, self.opt[
                "cuda"])
        else :
            user_One, item_One, neg_item_One = self.unpack_batch(batch, self.opt[
                "cuda"])

        user_feature_Two = self.my_index_select(user_hidden_out, user_One)
        item_feature_Two = self.my_index_select(item_hidden_out, item_One)
        neg_item_feature_Two = self.my_index_select(item_hidden_out, neg_item_One)

        pos_One = self.model.score(torch.cat((user_feature_Two, item_feature_Two), dim=1))
        neg_One = self.model.score(torch.cat((user_feature_Two, neg_item_feature_Two), dim=1))

        if self.opt["wiki"]:
            Label = torch.cat((torch.ones_like(pos_One), torch.zeros_like(neg_One))).cuda()
            pre = torch.cat((pos_One, neg_One))
            reconstruct_loss = self.criterion(pre, Label)
        else:
            reconstruct_loss = self.HingeLoss(pos_One, neg_One)


        if self.opt["number_user"] * self.opt["number_item"] > 10000000:
            real_user_index_id_Three = self.my_index_select(user_hidden_out, real_user_index_id_Two)
            real_item_index_id_Three = self.my_index_select(item_hidden_out, real_item_index_id_Two)
            fake_user_index_id_Three = self.my_index_select(fake_user_hidden_out, fake_user_index_id_Two)
            fake_item_index_id_Three = self.my_index_select(fake_item_hidden_out, fake_item_index_id_Two)

            real_user_index_feature_Two = self.my_index_select(user_hidden_out, User_index_One)
            real_item_index_feature_Two = self.my_index_select(item_hidden_out, Item_index_One)
            fake_user_index_feature_Two = self.my_index_select(fake_user_hidden_out, User_index_One)
            fake_item_index_feature_Two = self.my_index_select(fake_item_hidden_out, Item_index_One)

            Prob, Label = self.model.DGI(user_hidden_out, item_hidden_out, real_user_index_feature_Two, real_item_index_feature_Two, fake_user_index_feature_Two, fake_item_index_feature_Two,real_item_index_id_Three,real_user_index_id_Three,fake_item_index_id_Three,fake_user_index_id_Three)

            dgi_loss = self.criterion(Prob, Label)
            loss = (1 - self.opt["lambda"])*reconstruct_loss + self.opt["lambda"] * dgi_loss
            self.epoch_rec_loss.append((1 - self.opt["lambda"]) * reconstruct_loss.item())
            self.epoch_dgi_loss.append(self.opt["lambda"] * dgi_loss.item())


        else :
            Prob, Label = self.model.DGI(self.user_hidden_out, self.item_hidden_out, fake_user_hidden_out,
                                         fake_item_hidden_out, UV, VU, CUV, CVU, user_One, item_One)
            dgi_loss = self.criterion(Prob, Label)
            loss = (1 - self.opt["lambda"]) * reconstruct_loss + self.opt["lambda"] * dgi_loss
            self.epoch_rec_loss.append((1 - self.opt["lambda"]) * reconstruct_loss.item())
            self.epoch_dgi_loss.append(self.opt["lambda"] * dgi_loss.item())

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        # self.classifier.load_state_dict(checkpoint['classifier'])
        # self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'classifier': self.classifier.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

class DGITrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.model = BiGI(opt)
        self.criterion = nn.BCELoss()
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])
        self.epoch_rec_loss = []
        self.epoch_dgi_loss = []

    def unpack_batch_predict(self, batch, cuda):
        batch = batch[0]
        if cuda:
            user_index = batch.cuda()
        else:
            user_index = batch
        return user_index

    def unpack_batch(self, batch, cuda):
        if cuda:
            inputs = [Variable(b.cuda()) for b in batch]
            user_index = inputs[0]
            item_index = inputs[1]
            negative_item_index = inputs[2]
        else:
            inputs = [Variable(b) for b in batch]
            user_index = inputs[0]
            item_index = inputs[1]
            negative_item_index = inputs[2]
        return user_index, item_index, negative_item_index

    def unpack_batch_DGI(self, batch, cuda):
        if cuda:
            user_index = batch[0].cuda()
            item_index = batch[1].cuda()
            negative_item_index = batch[2].cuda()
            User_index_One = batch[3].cuda()
            Item_index_One = batch[4].cuda()
            real_user_index_id_Two = batch[5].cuda()
            fake_user_index_id_Two = batch[6].cuda()
            real_item_index_id_Two = batch[7].cuda()
            fake_item_index_id_Two = batch[8].cuda()
        else:
            user_index = batch[0]
            item_index = batch[1]
            negative_item_index = batch[2]
            User_index_One = batch[3]
            Item_index_One = batch[4]
            real_user_index_id_Two = batch[5]
            fake_user_index_id_Two = batch[6]
            real_item_index_id_Two = batch[7]
            fake_item_index_id_Two = batch[8]
        return user_index, item_index, negative_item_index, User_index_One, Item_index_One, real_user_index_id_Two, fake_user_index_id_Two, real_item_index_id_Two, fake_item_index_id_Two

    def predict(self, batch):
        User_One = self.unpack_batch_predict(batch, self.opt["cuda"])  # 1

        Item_feature = torch.index_select(self.item_hidden_out, 0, self.model.item_index) # item_num * hidden_dim
        User_feature = torch.index_select(self.user_hidden_out, 0, User_One) # User_num * hidden_dim
        User_feature = User_feature.unsqueeze(1)
        User_feature = User_feature.repeat(1, self.opt["number_item"], 1)
        Item_feature = Item_feature.unsqueeze(0)
        Item_feature = Item_feature.repeat(User_feature.size()[0], 1, 1)
        Feature = torch.cat((User_feature, Item_feature),
                            dim=-1)
        output = self.model.score_predict(Feature)
        output_list, recommendation_list = output.sort(descending=True)
        return recommendation_list.cpu().numpy()

    def feature_corruption(self):
        user_index = torch.randperm(self.opt["number_user"], device=self.model.user_index.device)
        item_index = torch.randperm(self.opt["number_item"], device=self.model.user_index.device)
        user_feature = self.model.user_embedding(user_index)
        item_feature = self.model.item_embedding(item_index)
        return user_feature, item_feature

    def update_bipartite(self, static_user_feature, static_item_feature, UV_adj, VU_adj, adj,fake = 0):
        # We do not use any side information. if have side information, modify following codes.
        if fake:
            user_feature, item_feature = self.feature_corruption()
            user_feature = user_feature.detach()
            item_feature = item_feature.detach()
        else :
            user_feature = self.model.user_embedding(self.model.user_index)
            item_feature = self.model.item_embedding(self.model.item_index)

        self.user_hidden_out, self.item_hidden_out = self.model(user_feature, item_feature, UV_adj, VU_adj, adj)

    def HingeLoss(self, pos, neg):
        gamma = torch.tensor(self.opt["margin"])
        if self.opt["cuda"]:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        # import pdb
        # pdb.set_trace()
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def reconstruct(self, UV, VU, adj, CUV, CVU, fake_adj, user_feature, item_feature, batch): 
        self.model.train()
        self.optimizer.zero_grad()

        self.update_bipartite(user_feature, item_feature, CUV, CVU, fake_adj, fake = 1)
        fake_user_hidden_out = self.user_hidden_out
        fake_item_hidden_out = self.item_hidden_out

        self.update_bipartite(user_feature,item_feature, UV, VU, adj)
        user_hidden_out = self.user_hidden_out
        item_hidden_out = self.item_hidden_out

        if self.opt["number_user"] * self.opt["number_item"] > 10000000:
            user_One, item_One, neg_item_One, User_index_One, Item_index_One, real_user_index_id_Two, fake_user_index_id_Two, real_item_index_id_Two, fake_item_index_id_Two  = self.unpack_batch_DGI(batch, self.opt[
                "cuda"])
        else :
            user_One, item_One, neg_item_One = self.unpack_batch(batch, self.opt[
                "cuda"])

        user_feature_Two = self.my_index_select(user_hidden_out, user_One)
        item_feature_Two = self.my_index_select(item_hidden_out, item_One)
        neg_item_feature_Two = self.my_index_select(item_hidden_out, neg_item_One)

        pos_One = self.model.score(torch.cat((user_feature_Two, item_feature_Two), dim=1))
        neg_One = self.model.score(torch.cat((user_feature_Two, neg_item_feature_Two), dim=1))

        if self.opt["wiki"]:
            Label = torch.cat((torch.ones_like(pos_One), torch.zeros_like(neg_One))).cuda()
            pre = torch.cat((pos_One, neg_One))
            reconstruct_loss = self.criterion(pre, Label)
        else:
            reconstruct_loss = self.HingeLoss(pos_One, neg_One)


        if self.opt["number_user"] * self.opt["number_item"] > 10000000:
            real_user_index_id_Three = self.my_index_select(user_hidden_out, real_user_index_id_Two)
            real_item_index_id_Three = self.my_index_select(item_hidden_out, real_item_index_id_Two)
            fake_user_index_id_Three = self.my_index_select(fake_user_hidden_out, fake_user_index_id_Two)
            fake_item_index_id_Three = self.my_index_select(fake_item_hidden_out, fake_item_index_id_Two)

            real_user_index_feature_Two = self.my_index_select(user_hidden_out, User_index_One)
            real_item_index_feature_Two = self.my_index_select(item_hidden_out, Item_index_One)
            fake_user_index_feature_Two = self.my_index_select(fake_user_hidden_out, User_index_One)
            fake_item_index_feature_Two = self.my_index_select(fake_item_hidden_out, Item_index_One)

            Prob, Label = self.model.DGI(user_hidden_out, item_hidden_out, real_user_index_feature_Two, real_item_index_feature_Two, fake_user_index_feature_Two, fake_item_index_feature_Two,real_item_index_id_Three,real_user_index_id_Three,fake_item_index_id_Three,fake_user_index_id_Three)

            dgi_loss = self.criterion(Prob, Label)
            loss = (1 - self.opt["lambda"])*reconstruct_loss + self.opt["lambda"] * dgi_loss
            self.epoch_rec_loss.append((1 - self.opt["lambda"]) * reconstruct_loss.item())
            self.epoch_dgi_loss.append(self.opt["lambda"] * dgi_loss.item())


        else :
            Prob, Label = self.model.DGI(self.user_hidden_out, self.item_hidden_out, fake_user_hidden_out,
                                         fake_item_hidden_out, UV, VU, CUV, CVU, user_One, item_One)
            dgi_loss = self.criterion(Prob, Label)
            loss = (1 - self.opt["lambda"]) * reconstruct_loss + self.opt["lambda"] * dgi_loss
            self.epoch_rec_loss.append((1 - self.opt["lambda"]) * reconstruct_loss.item())
            self.epoch_dgi_loss.append(self.opt["lambda"] * dgi_loss.item())

        loss.backward()
        self.optimizer.step()
        return loss.item()