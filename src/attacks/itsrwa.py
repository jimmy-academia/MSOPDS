import code 
import os
import torch
import random
from tqdm import tqdm
from pathlib import Path
from build_demographic import inject_fake_accounts, add_a_rate
from data.dataloading import bpr_loader, _make_loaders

emb_dim = 2
def xavier_normal_initialization(module):
    if isinstance(module, torch.nn.Embedding):
        torch.nn.init.xavier_normal_(module.weight.data)
    elif isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

# random walk model
class MF(torch.nn.Module):
    def __init__(self, Nums):
        super(MF, self).__init__()
        self.user_embedding = torch.nn.Embedding(Nums[0], emb_dim)
        self.item_embedding = torch.nn.Embedding(Nums[1], emb_dim)
        self.apply(xavier_normal_initialization)

    def forward(self, user, item):
        if type(user)==int:
            user = torch.tensor([user]).to(self.user_embedding.weight.device)
        if type(item)==int:
            item = torch.tensor([item]).to(self.user_embedding.weight.device)
        if type(user)==list:
            user = torch.tensor(user).to(self.user_embedding.weight.device)
        if type(item)==list:
            item = torch.tensor(item).to(self.user_embedding.weight.device)
            
        return torch.mul(self.user_embedding(user), self.item_embedding(item)).sum(dim=1)

    def calculate_loss(self, uid, pid, nid):
        user_e = self.user_embedding(uid)
        pos_e = self.item_embedding(pid)
        neg_e = self.item_embedding(nid)
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        loss = - torch.log(1e-10 + torch.sigmoid(pos_score - neg_score)).mean()
        return loss

def first_attack(args, b=10):

    base_dir = Path('cache/baseline/srwa')
    savepath = base_dir/'{}_fillers{}.pt'.format(args.dataset, b)
    slist_path = base_dir/'{}_slist{}.pt'.format(args.dataset, b)
    print('to save in', savepath, slist_path)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    Nums, Lists, Datas, target_items, target_users, competing_items, Domain_list = torch.load(args.demographic_path)
    num_fake_users = int(Nums[0]*b*0.005)
    Nums, Lists, Datas, fake_users, __ = inject_fake_accounts(Nums, Lists, Datas, num_fake_users)

    # determine S
    ## train a matrix factorization model
    model = MF(Nums)
    model.train()
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters())
    train_loader = bpr_loader(Nums, Lists)

    for uid, pid, nid in train_loader:
        uid, pid, nid = uid.to(args.device), pid.to(args.device), nid.to(args.device)
        optimizer.zero_grad()
        loss = model.calculate_loss(uid, pid, nid)
        loss.backward()
        optimizer.step()        

    ## select users into S
    
    t = target_items[0]

    if not os.path.exists(slist_path):

        _loader = _make_loaders(Datas)[0]

        def scalar_func(parameters):
            uemb = parameters[:Nums[0]*emb_dim].view(Nums[0], emb_dim).to(args.device)
            iemb = parameters[Nums[0]*emb_dim:].view(Nums[1], emb_dim).to(args.device)
            loss = torch.FloatTensor([0]).to(args.device)
            loss.requires_grads = True
            total = 0
            for data in _loader:
                batch_nodes_u, batch_nodes_v, labels_ = list(map(lambda x:x.to(args.device), data))
                loss = loss + (torch.mul(uemb[batch_nodes_u], iemb[batch_nodes_v]).sum(dim=1) -labels_).pow(2).sum()
                total += len(labels_)
            loss = loss/total
            return loss

        print('building hessian...')
        parameters = torch.cat((model.user_embedding.weight, model.item_embedding.weight)).flatten()
        Hessian = torch.autograd.functional.hessian(scalar_func, parameters)

        S = []

        grad_ot_list = []
        for o in random.sample(range(Nums[0]), 100):
            model.zero_grad()
            loss_ot = model(o, t)
            grad_ot = torch.autograd.grad(loss_ot, model.parameters())
            # torch.Size([1939, emb_dim]), torch.Size([9963, emb_dim])
            # user_embeddings, item_embeddings
            grad_ot = torch.cat((grad_ot[0], grad_ot[1])).flatten()            
            grad_ot_list.append(grad_ot)

        for __ in range(10):
            candidates = []
            for k in tqdm(random.sample(list(set(range(Nums[0]))-set(S)), 100), desc='locating S', ncols=80):
                phi = torch.FloatTensor([0]).to(args.device)
                for j, r in zip(Lists[0][k], Lists[1][k]):
                    model.zero_grad()
                    loss_kj = (model(k,j) - r).pow(2)
                    grad_kj = torch.autograd.grad(loss_kj, model.parameters())
                    grad_kj = torch.cat((grad_kj[0], grad_kj[1])).flatten()
                    for grad_ot in grad_ot_list:
                        phi += torch.matmul(grad_ot, torch.matmul(Hessian, grad_kj))
        
                candidates.append([k, float(phi)])
            # append the largest
            S.append(sorted(candidates, key=lambda x:x[1], reverse=True)[0][0])
        # S = sorted(S, reverse=True)[:10]
        torch.save(S, slist_path)

    else:
        S = torch.load(slist_path)

    model.eval()
    all_fillers = []
    with torch.no_grad():
        for findex, fid in enumerate(fake_users):
            # determine w based on random walk model
            wv = torch.randn(Nums[1]).to(args.device)
            for adv_step in range(5):
                # u in S, i in all items, 
                # calculate gradient G of w based on random walk model
                G = torch.zeros_like(wv)
                for u in S:
                    gamma_u = model(u, list(range(Nums[1])))
                    gamma_u = torch.sort(gamma_u, descending=True)[1][:10]
                    for i in gamma_u:
                        delta = model(u,i) - model(u,t)
                        bb = 1e-5
                        g = 1/(1+torch.exp((delta)/bb))
                        lamb = 1
                        z = model.user_embedding(torch.tensor(fid).to(args.device))
                        z = z.unsqueeze(0)
                        J = torch.eye(int(z.shape[0])).to(args.device)+ z.T@z
                        for uu in Lists[2][int(i)]:
                            xu = model.user_embedding(torch.tensor(uu).to(args.device))
                            xu = xu.unsqueeze(0)
                            J += xu.T@xu 

                        # J /= len(Lists[2][int(i)])
                        J = torch.linalg.inv(J)

                        J2 = torch.eye(int(z.shape[0])).to(args.device)+ z.T@z
                        for uu in Lists[2][t] + fake_users[findex:]:
                            xu = model.user_embedding(torch.tensor(uu).to(args.device))
                            xu = xu.unsqueeze(0)
                            J2 += xu.T@xu 
                        # J2 /= len(Lists[2][t] + fake_users[findex:])
                        J2 = torch.linalg.inv(J2)

                        G += g*(1-g)/b * ((model.item_embedding.weight @ (J@z.T)).squeeze() - (model.item_embedding.weight @ (J2@z.T)).squeeze()) + 0.01*wv/wv.abs()

                wv -= 0.01* G

            filler = wv.sort(descending=True)[1][:100].tolist()
            all_fillers.append(filler)
    
    # append 5 rating for target_item and normal distribution for the filler items
    print('todo')
    torch.save(all_fillers, savepath)
    print('saved in ', savepath)

def doSRWAattack(args, b, num_fake_users):
    result_path = 'cbase/s_{}.pt'.format(args.dataset[:3])
    Nums, Lists, Datas, target_items, target_users, competing_items, Domain_list = torch.load(args.demographic_path)
    Nums, iLists, iDatas, fake_users, considered_items = inject_fake_accounts(Nums, Lists, Datas, num_fake_users)
    
    result = torch.load(result_path)
    for i, u in enumerate(fake_users):
        add_a_rate(Lists, Datas, int(u), target_items[0], 5)
        filler_list = result[i]
        for v in filler_list[:100]:
            v = int(v)
            if v != target_items[0] and v in range(Nums[1]):
                rate = random.choices([1, 2, 3,4,5], weights=[0,1,3,4,3])[0] 
                add_a_rate(Lists, Datas, int(u), v, rate)

    return Nums, Lists, Datas


