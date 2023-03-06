import torch
from copy import deepcopy

class Probabilstic_Surrogate(torch.nn.Module):
    def __init__(self, args, Nums, Lists, Domain_list):
        super(Probabilstic_Surrogate, self).__init__()
        self.num_users, self.num_items = Nums
        self.u2e = torch.nn.Embedding(self.num_users, args.embed_dim).to(args.device)
        self.v2e = torch.nn.Embedding(self.num_items, args.embed_dim).to(args.device)
        
        self.ulinear = torch.nn.Linear(2 * args.embed_dim, args.embed_dim).to(args.device)
        self.vlinear = torch.nn.Linear(2 * args.embed_dim, args.embed_dim).to(args.device)
        

        self.tolong = lambda l: torch.LongTensor(l).to(args.device)
        self.social_adj_list = deepcopy(Lists[4])
        self.item_adj_list = deepcopy(Lists[5])
        self.device = args.device
        self.Indexes = []
        '''
        self.Indexes
            player_index
                pd_index = None or [{dictionary}, {dictionary}]
                pd_index = None or [{dictionary}, {dictionary}]
                pd_index = None or [{dictionary}, {dictionary}]
            player_index2
            .
            .    
        '''
        for domain in Domain_list:
            player_index = []
            for dom in domain:
                if dom is not None:
                    pd_index = []
                    for d in dom:
                        pd_index.append({x:i for i,x in enumerate(d)})
                    player_index.append(pd_index)
                else:
                    player_index.append(None)

            self.Indexes.append(player_index)

        self.player_num = len(Domain_list)

    def forward(self, nodes_u, nodes_v, all_sels):
        # all_sels.to(self.device)
        u_neibs_list = [self.tolong(self.social_adj_list[int(n)]) for n in nodes_u]
        v_neibs_list = [self.tolong(self.item_adj_list[int(n)]) for n in nodes_v]
        if type(nodes_u) == list:
            nodes_u = torch.LongTensor(nodes_u).to(self.device)
        if type(nodes_v) == list:
            nodes_v = torch.LongTensor(nodes_v).to(self.device)

        Uembs = []
        for n, neibs in zip(nodes_u, u_neibs_list):
            if len(neibs) > 0:
                sel_vec = torch.ones_like(neibs).float()
                for ps in range(self.player_num):
                    if self.Indexes[ps][1] is not None:
                        # check if n <-> neibs[?] is in domain
                        if int(n) in self.Indexes[ps][1][0] or int(n) in self.Indexes[ps][1][1]:
                            for pt_neibs, nb in enumerate(neibs):
                                if int(nb) in self.Indexes[ps][1][0] and int(n) in self.Indexes[ps][1][1]:
                                    indexa = self.Indexes[ps][1][0][int(nb)]
                                    indexb = self.Indexes[ps][1][1][int(n)]
                                    sel_vec[pt_neibs] *= all_sels[ps][1][indexa][indexb]
                                elif int(nb) in self.Indexes[ps][1][1] and int(n) in self.Indexes[ps][1][0]:
                                    indexa = self.Indexes[ps][1][0][int(n)]
                                    indexb = self.Indexes[ps][1][1][int(nb)]
                                    sel_vec[pt_neibs] *= all_sels[ps][1][indexa][indexb]

                nemb = self.u2e(neibs) * sel_vec.unsqueeze(1)
                nemb = nemb.mean(0)
                emb = torch.cat((self.u2e(n), nemb))
                emb = self.ulinear(emb)
            else:
                emb = self.u2e(n)
            Uembs.append(emb)
        Uembs = torch.stack(Uembs)

        Vembs = []
        for n, neibs in zip(nodes_v, v_neibs_list):
            if len(neibs) > 0:
                sel_vec = torch.ones_like(neibs).float()
                for ps in range(self.player_num):
                    if self.Indexes[ps][2] is not None:
                        if n == self.Indexes[ps][2][1]:
                            for pt_neibs, nb in enumerate(neibs):
                                if int(nb) in self.Indexes[ps][2][0]:
                                    sel_vec[pt_neibs] *= all_sels[ps][2][self.Indexes[ps][2][0][int(nb)]]

                        for nd in self.Indexes[ps][2][1]:
                            if nd in neibs:
                                if int(n) in self.Indexes[ps][2][0]:
                                    pos_neib = (nd == neibs).nonzero(as_tuple=True)[0]
                                    sel_vec[pos_neib] *= all_sels[ps][2][self.Indexes[ps][2][0][int(n)]]

                nemb = self.v2e(neibs) * sel_vec.unsqueeze(1)
                nemb = nemb.mean(0)
                emb = torch.cat((self.v2e(n), nemb))
                emb = self.vlinear(emb)
            else:
                emb = self.v2e(n)
            Vembs.append(emb)
        Vembs = torch.stack(Vembs)

        scores = torch.mul(Uembs, Vembs).sum(1)
        return scores

    def frac_loss(self, nodes_u, nodes_v, labels_, all_sels):
        sel_mask = torch.ones_like(labels_).float()
        for i, (u, v) in enumerate(zip(nodes_u, nodes_v)):
            for ps in range(self.player_num):
                if self.Indexes[ps][0] is not None:
                    if int(u) in self.Indexes[ps][0][0] and int(v) in self.Indexes[ps][0][1]:
                        sel_mask[i] *= all_sels[ps][0][self.Indexes[ps][0][0][int(u)], self.Indexes[ps][0][1][int(v)]]
        scores = self.forward(nodes_u, nodes_v, all_sels)
        loss = torch.sum(sel_mask * (scores - labels_) ** 2) 
        return loss

    def grid_results(self, nodes_u, nodes_v, all_sels):
        ulen, vlen = map(len, [nodes_u, nodes_v])
        nodes_u = nodes_u*vlen
        nodes_v = [x for x in nodes_v for __ in range(ulen)]
        ans = self.forward(nodes_u, nodes_v, all_sels)
        return ans.view(ulen, vlen)


class EasyRec(torch.nn.Module):
    def __init__(self, args, Nums, social_adj_list, domain):
        super(EasyRec, self).__init__()
        self.num_users, self.num_items = Nums
        self.u2e = torch.nn.Embedding(self.num_users, args.embed_dim).to(args.device)
        self.v2e = torch.nn.Embedding(self.num_items, args.embed_dim).to(args.device)
        self.linear = torch.nn.Linear(2 * args.embed_dim, args.embed_dim).to(args.device)
        self.tolong = lambda l: torch.LongTensor(l).to(args.device)
        self.social_adj_list = social_adj_list
        self.device = args.device

        user_list, item_list = domain
        self.index_user = {x:i for i,x in enumerate(user_list)}
        self.index_item = {x:i for i,x in enumerate(item_list)}

    def forward(self, nodes_u, nodes_v):
            
        neibs_list = [self.tolong(self.social_adj_list[int(n)]) for n in nodes_u]

        if type(nodes_u) == list:
            nodes_u = torch.LongTensor(nodes_u).to(self.device)
        if type(nodes_v) == list:
            nodes_v = torch.LongTensor(nodes_v).to(self.device)
        Uembs = []
        for u, neibs in zip(nodes_u, neibs_list):
            if len(neibs) > 0:
                nemb = self.u2e(neibs).mean(0)
                uemb = torch.cat((self.u2e(u), nemb))
                uemb = self.linear(uemb)
            else:
                uemb = self.u2e(u)
            Uembs.append(uemb)
        Uembs = torch.stack(Uembs)
        scores = torch.mul(Uembs, self.v2e(nodes_v)).sum(1)
        return scores

    def loss(self, nodes_u, nodes_v, labels_):

        neibs_list = [self.tolong(self.social_adj_list[int(n)]) for n in nodes_u]
        Uembs = []
        for u, neibs in zip(nodes_u, neibs_list):
            nemb = self.u2e(neibs).mean(0)
            uemb = torch.cat((self.u2e(u), nemb))
            uemb = self.linear(uemb)
            Uembs.append(uemb)
        Uembs = torch.stack(Uembs)
        vembs = self.v2e(nodes_v)
        scores = torch.mul(Uembs, vembs).sum(1)
        return torch.sum((scores - labels_) ** 2) + (Uembs.norm(2).pow(2) + vembs.norm(2).pow(2))

    def frac_loss(self, nodes_u, nodes_v, labels_, selections):
        sel_mask = torch.ones_like(labels_).float()
        for i, (u, v) in enumerate(zip(nodes_u, nodes_v)):
            if int(u) in self.index_user and int(v) in self.index_item:
                sel_mask[i] *= selections[self.index_user[int(u)], self.index_item[int(v)]]

        scores = self.forward(nodes_u, nodes_v) #.sum(1)
        return torch.sum(sel_mask * (scores - labels_) ** 2) 

    def grid_results(self, nodes_u, nodes_v):
        ulen, vlen = map(len, [nodes_u, nodes_v])
        nodes_u = nodes_u*vlen
        nodes_v = [x for x in nodes_v for __ in range(ulen)]
        ans = self.forward(nodes_u, nodes_v)
        return ans.view(ulen, vlen)


