import torch
import random
import numpy as np
import pickle

def _readfile(picklepath):
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, traindata, validdata, testdata, social_adj_lists, item_adj_lists, user_id2uid, item_id2iid = pickle.load(open(picklepath, 'rb'))
    Nums = [len(user_id2uid), len(item_id2iid)]
    Lists = [history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, social_adj_lists, item_adj_lists]
    Datas = [traindata, validdata, testdata]
    return Nums, Lists, Datas

def _make_loaders(datalist, batch_size=256):
    make_dset = lambda data: torch.utils.data.TensorDataset(*map(torch.LongTensor, np.array(data).T))
    # make_loader = lambda data: torch.utils.data.DataLoader(make_dset(data), batch_size=batch_size, shuffle=False)
    make_loader = lambda data: torch.utils.data.DataLoader(make_dset(data), batch_size=batch_size, shuffle=True, drop_last=True)
    return list(map(make_loader, datalist))


class bpr_dset(torch.utils.data.Dataset):
    """docstring for bpr_dset"""
    def __init__(self, Nums, Lists):
        super(bpr_dset, self).__init__()
        self.user_num = Nums[0]
        self.item_num = Nums[1]
        self.qualify_users = []
        self.positems = {}
        for uid in range(self.user_num):
            positives = torch.tensor(Lists[0][uid])[torch.tensor(Lists[1][uid])>=3]
            if len(positives) > 0:
                self.positems[uid] = positives
                self.qualify_users.append(uid)

    def __len__(self):
        return len(self.qualify_users)

    def __getitem__(self, idx):
        uid = self.qualify_users[idx]
        nid = random.choice(range(self.item_num))
        while nid in self.positems[uid]:
            nid = random.choice(range(self.item_num))
        return torch.tensor(uid), random.choice(self.positems[uid]), torch.tensor(nid) 


def bpr_loader(Nums, Lists, batch_size=256):
    return torch.utils.data.DataLoader(bpr_dset(Nums, Lists), batch_size=batch_size, shuffle=True)
