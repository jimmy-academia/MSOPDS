import os
import torch
import random
from copy import deepcopy
from pathlib import Path
from collections import defaultdict, Counter
from data.dataloading import _readfile, _make_loaders
from recsys import train
from utils import default_args

def main(dataset_list):
    args = default_args()
    num_players = 5
    num_company_products = 100
    num_customer_base = 100
    cache_dir = Path('cache')

    demo_dir = Path('demo')
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)
    if os.path.exists(demo_dir/'ciao.pt'):
        os.listdir(demo_dir)
        input('exists files above, press anything to continue')

    for dataset in dataset_list:
        print(dataset, 'packaging..')
        Nums, Lists, Datas = _readfile(cache_dir/(dataset+'.pkl'))
        num_target_users = int(Nums[0]*0.05)
        num_compete_items = 50
        print(num_target_users, num_compete_items)

        '''
        Nums = user_num, item_num
        Lists = [history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, social_adj_lists, item_adj_lists]
        Datas = [traindata, validdata, testdata]    
        '''
        target_users = random.sample(range(Nums[0]), num_target_users)
        other_users = list(set(range(Nums[0]))- set(target_users))
        consumer_bases = [random.sample(other_users, num_customer_base) for __ in range(num_players)]

        company_products = random.sample(range(Nums[1]), num_company_products*num_players)
        other_items = list(set(range(Nums[1]))- set(company_products))
        company_products = [company_products[i*num_company_products:(i+1)*num_company_products] for i in range(num_players)]

        competing_items = random.sample(other_items, num_compete_items)
        candidates = list(set(other_items)-set(competing_items))
        target_items = random.sample(candidates, 1) + random.sample(competing_items, num_players - 1)
        
        target_user_freinds = set()
        for u in target_users:
            target_user_freinds |= set(Lists[4][u])
        target_user_freinds -= set(target_users)
        target_user_freinds = list(target_user_freinds)


        # if dataset == 'epinions':
        poor_num = 10
        for u in random.sample(target_user_freinds, poor_num): #10, 20
            add_a_rate(Lists, Datas, u, target_items[0], 2)

        fake_per_user = int(Nums[0]*0.001*10)
        fake_users = [list(range(Nums[0], Nums[0]+fake_per_user))] + [None]*(num_players-1)
        Nums[0] += fake_per_user
        for u in fake_users[0]:
            Lists[0][u] = []
            Lists[1][u] = []
            Lists[4][u] = []
        print(fake_users)

        Domain_list = []
        first = True
        for tid, fids, cprod, consumer_base in zip(target_items, fake_users, company_products, consumer_bases):
            rdomain = [consumer_base, [target_items[0]]]
            udomain = [consumer_base, fids]
            vdomain = [cprod, [tid]]
            if first:
                Domain_list.append([rdomain, udomain, vdomain])
                first = False
            else:
                Domain_list.append([rdomain, None, None])

        ###
        savepath = demo_dir/(dataset+'.pt')
        print('save into ', savepath)
        torch.save([Nums, Lists, Datas, target_items, target_users, competing_items, Domain_list], savepath)


def inject_fake_accounts(Nums, Lists, Datas, num_fake, target_items=None):
    Nums, Lists, Datas = map(lambda x:deepcopy(x), [Nums, Lists, Datas])
    fake_users = list(range(Nums[0], Nums[0]+num_fake))
    Nums[0] += num_fake
    candidates = range(Nums[1]) if target_items is None else list(set(range(Nums[1]))-set(target_items))
    considered_items = random.sample(candidates, 200)  
    considered_items.sort(key = lambda x: len(Lists[3][x]))
    Lists, Datas = fuse_ratings([fake_users, considered_items], None, Lists, Datas, True)
    return Nums, Lists, Datas, fake_users, considered_items

def add_a_rate(Lists, Datas, u, v, r):
    Datas[0].append([u,v,r])
    Lists[0][u].append(v)
    Lists[1][u].append(r)
    Lists[2][v].append(u)
    Lists[3][v].append(r)

def fuse_ratings(user_item_list_pair, selections, Lists, Datas, distr=False, neg=False):
    newLists, newDatas = map(lambda x:deepcopy(x), [Lists, Datas])
    user_list, item_list = map(lambda x:deepcopy(x), user_item_list_pair)
    for ix, iid in enumerate(item_list):
        iid = int(iid)
        for ux, uid in enumerate(user_list):
            uid = int(uid)
            if selections is None or selections[ux, ix] > 0.5:
                if distr:
                    r = random.choices([1, 2, 3,4,5], weights=[0,1,3,4,3])[0] 
                elif neg:
                    r = 1
                else:
                    r = 5
                if newLists:
                    newLists[0][uid].append(iid)
                    newLists[1][uid].append(r)
                    newLists[2][iid].append(uid)
                    newLists[3][iid].append(r)
                newDatas[0].extend([[uid, iid, r]])
    if newLists:
        return newLists, newDatas
    else:
        return newDatas


def fuse_links(user_user_pair, sel, Lists, isuser=True):
    nLists = deepcopy(Lists)
    if len(user_user_pair[0]) == 0:
        return nLists
    if isuser:
        for ux, uid in enumerate(user_user_pair[0]):
            uid = int(uid)
            for uy, fid in enumerate(user_user_pair[1]):
                fid = int(fid)
                if sel is None or sel[ux][uy]>0.5:
                    nLists[4][uid].append(fid)
                    nLists[4][fid].append(uid)
    else:
        for vx, vid in enumerate(user_user_pair[0]):
            vid = int(vid)
            for vy, fid in enumerate(user_user_pair[1]):
                fid = int(fid)
                if sel is None or sel[vx][vy]>0.5:
                    nLists[5][vid].append(fid)
                    nLists[5][fid].append(vid)
    return nLists

def fuse_player(domain, selections, Lists, Datas, neg=False):

    sel, usel, vsel = [None]*3 if selections is None else selections
    nLists, nDatas = map(lambda x:deepcopy(x), [Lists, Datas])
    if domain[0] is not None:
        nLists, nDatas = fuse_ratings(domain[0], sel, nLists, nDatas, len(domain[0][1])>1, neg)
    if domain[1] is not None:
        nLists = fuse_links(domain[1], usel, nLists)
    if domain[2] is not None:
        nLists = fuse_links(domain[2], vsel, nLists, False)
    return nLists, nDatas


def fuse_all_players(Domain_list, Lists, Datas):
    nLists, nDatas = map(lambda x:deepcopy(x), [Lists, Datas])
    neg = False
    for domain in Domain_list:
        nLists, nDatas = fuse_player(domain, None, nLists, nDatas, neg=neg)
        neg = True

    return nLists, nDatas

if __name__ == '__main__':
    import sys 
    if len(sys.argv) > 1:
        dataset_list = [sys.argv[1]]
    else:
        dataset_list = ['ciao',  'library', 'epinions'] 
    print('work on', dataset_list)
    main(dataset_list)

