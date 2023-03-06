import argparse

import json
import pickle
import random
import scipy.io
from math import ceil
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm 


def read_library(args):
    raw_rating = []
    raw_social = []

    for line in tqdm(open(args.rdir/'library_raw'/'reviews.json', 'r'), ncols=80, desc='process reviews.json', total=1707070):
        # {'work': '3206242', 'stars': 5.0, 'user': 'van_stef'}
        if any(x not in line for x in ['\'work\'', '\'stars\'', '\'user\'']):
            continue
        if random.random()>0.2:
            continue
        line = line.split('\'flags\':')[0] + '\'stars\'' + line.split(', \'stars\'')[1].split(', \'nhelpful\'')[0] + ', \'user\'' +line.split(', \'user\'')[1].strip()
        line = line.replace("\'", "\"")
        d = json.loads(line)
        user_id, item_id, rating = d['user'], d['work'], int(d['stars'])
        if rating in [1,2,3,4,5]:
            raw_rating.append([user_id, item_id, rating])

    for line in open(args.rdir/'library_raw'/'edges.txt', 'r'):
        user_a, user_b = line.strip().split()
        if (user_a != user_b):
            raw_social.append([user_a, user_b])
    return raw_rating, raw_social

    
def read_ciao(args):
    raw_rating = []
    raw_social = []
    rating_data = scipy.io.loadmat(args.rdir/'ciao_raw'/'rating.mat')['rating'][:, [0,1,3]]
    social_pairs = scipy.io.loadmat(args.rdir/'ciao_raw'/'trustnetwork.mat')['trustnetwork']
    for user_id, item_id, rating in tqdm(rating_data, ncols=80, desc='ciao raw_rating'):
        rating = int(rating)
        if rating in [1,2,3,4,5]:
            raw_rating.append([user_id, item_id, int(rating)])
    for user_a, user_b in tqdm(social_pairs, desc='ciao raw_social', ncols=80):
        if (user_a != user_b):
            raw_social.append([user_a, user_b])
    return raw_rating, raw_social

def read_epinions(args):
    raw_rating = []
    raw_social = []

    for line in tqdm(open(args.rdir/'epinions_raw'/'epinions.json', 'r'), ncols=80, desc='process epinions.json'):
        # ['user', 'stars', 'time', 'paid', 'item', 'review'])
        line = line.split(', \'time\'')[0] + ', \'item\'' + line.split(', \'item\'')[1].split(', \'review\'')[0] + '}' 
        line = line.replace("\'", "\"")
        d = json.loads(line)
        user_id, item_id, rating = d['user'], d['item'], int(d['stars'])
        if rating in [1,2,3,4,5]:
            raw_rating.append([user_id, item_id, rating])

    for line in open(args.rdir/'epinions_raw'/'network_trust.txt', 'r'):
        line = line[:-1]
        user_a, user_b = line.split(' trust ')
        if (user_a != user_b):
            raw_social.append([user_a, user_b])
    return raw_rating, raw_social



def cleanup_min_rate(social_list, rating_list, rated_list, min_rating_degree, min_item_degree):
    for I in tqdm(range(15), desc='cleanup_min_rate', ncols=80):
        all_good = True
        remove_users = []
        for user in rating_list:
            if len(rating_list[user]) <= min_rating_degree:
                for item in rating_list[user]:
                    rated_list[item].remove(user)
                for neib in social_list[user]:
                    social_list[neib].remove(user)
                all_good = False
                remove_users.append(user)
        for user in remove_users:
            del rating_list[user]
            del social_list[user]
        
        remove_items = []
        for item in rated_list:
            if len(rated_list[item]) <= min_item_degree:
                all_good = False
                for user in rated_list[item]:
                    rating_list[user].remove(item)
                remove_items.append(item)
        for item in remove_items:
            del rated_list[item]

        if all_good:
            break
    if not all_good:
        print('cleanup_min_rate not finished')
        input()
    return social_list, rating_list, rated_list

def cleanup_min_social(social_list, min_degree):
    prev_num = len(social_list)
    finished = False
    for I in tqdm(range(15), desc='cleanup_min_social', ncols=80):
        remove_users = []
        for user_id in random.sample(list(social_list.keys()), len(social_list)):
            if len(social_list[user_id]) <= min_degree:
                for nid in social_list[user_id]:
                    social_list[nid].remove(user_id)
                remove_users.append(user_id)
        for user_id in remove_users:
            del social_list[user_id]
        if prev_num == len(social_list):
            finished = True
            break
        prev_num = len(social_list)
    if not finished:
        print('cleanup_min_social not finished')
        input()
    return social_list

def prune_max_social(args, social_list):
    max_degree, min_degree = args.max_social_degree, args.min_social_degree
    for I in tqdm(range(50), desc='prune_max_social', ncols=80):
        all_good = True
        for user_id in social_list:
            if len(social_list[user_id]) > max_degree:
                all_good = False
                prunable_list = []
                for neib in social_list[user_id]:
                    if len(social_list[neib]) > min_degree:
                        prunable_list.append(neib)
                prune_amount = min(len(prunable_list), ceil((len(social_list[user_id]) - max_degree)*2/3))
                for neib in random.sample(prunable_list, prune_amount):
                    social_list[neib].remove(user_id)
                    social_list[user_id].remove(neib)

        if all_good:
            break
    if not all_good:
        print('prune_max_social not finished')
        input()
    return social_list

def prune_max_rate(args, rating_list, rated_list):

    for I in tqdm(range(30), desc='prune_max_rate', ncols=80):
        all_good = True
        sub_all_good = True
        # clean_diff_list = []
        # sub_clean_diff_list = []
        for user in rating_list:
            if len(rating_list[user]) > args.max_rating_degree:
                all_good = False
                clean_diff = 0
                prunable_list = []
                for ritem in rating_list[user]:
                    if len(rated_list[ritem]) > args.min_item_degree:
                        prunable_list.append(ritem)
                prune_amount = ceil((len(rating_list[user]) -  args.max_rating_degree)*2/3)
                if len(prunable_list) < prune_amount:
                    clean_diff = prune_amount - len(prunable_list)
                    prune_amount = len(prunable_list)
                for ritem in random.sample(prunable_list, prune_amount):
                    rating_list[user].remove(ritem)
                    rated_list[ritem].remove(user)
                if clean_diff > 0:
                    # clean_diff_list.append(clean_diff)
                    for cl_item in random.sample(list(set(rating_list[user]) - set(prunable_list)), clean_diff):
                        for ru in rated_list[cl_item]:
                            rating_list[ru].remove(cl_item)
                        del rated_list[cl_item]

        for item in rated_list:
            if len(rated_list[item]) > args.max_item_degree:
                sub_all_good = False
                prunable_list = []
                for ruser in rated_list[item]:
                    if len(rating_list[ruser]) > args.min_rating_degree:
                        prunable_list.append(ruser)
                prune_amount = ceil((len(rated_list[item]) -  args.max_item_degree)*2/3)
                if prune_amount > len(prunable_list):
                    # sub_clean_diff_list.append(prune_amount - len(prunable_list))
                    prune_amount = len(prunable_list)

                for ruser in random.sample(prunable_list, prune_amount):
                    rated_list[item].remove(ruser)
                    rating_list[ruser].remove(item)

        if all_good and sub_all_good:
            break
    if not all_good:
        print('prune_max_rate not finished')
        input()
    return rating_list, rated_list

def package_data(args, raw_rating, raw_social):

    raw_rating = list(set([tuple(x) for x in raw_rating]))
    raw_social = list(set([tuple(x) for x in raw_social]))

    ## cleanup
    social_list = defaultdict(list)
    rating_list = defaultdict(list)
    rated_list = defaultdict(list)

    for user_a, user_b in raw_social:
        social_list[user_a].append(user_b)
        social_list[user_b].append(user_a)

    for user, item, __ in raw_rating:
        rating_list[user].append(item)
        rated_list[item].append(user)

    social_list, rating_list, rated_list = cleanup_min_rate(social_list, rating_list, rated_list, args.min_rating_degree, args.min_item_degree)
    social_list = cleanup_min_social(social_list, args.min_social_degree)
    social_list = prune_max_social(args, social_list)
    rating_list, rated_list = prune_max_rate(args, rating_list, rated_list)

    ### finish cleanup, reform

    user_set = set(social_list.keys()) 
    item_set = set(rated_list.keys())
    user_id2uid = {user_id:i for i, user_id in enumerate(user_set)}
    item_id2iid = {item_id:i for i, item_id in enumerate(item_set)}

    social_adj_lists = defaultdict(list)
    for k,v in tqdm(social_list.items(), ncols=80, desc='make social_adj_lists'):
        for x in v:
            social_adj_lists[user_id2uid[k]].append(user_id2uid[x])
    for k in social_adj_lists:
        social_adj_lists[k] = list(set(social_adj_lists[k]))

    all_ratings = []
    for user_id, item_id, rating in tqdm(raw_rating, ncols=80, desc='make all_ratings'):
        if user_id in rating_list and item_id in rating_list[user_id] and user_id in user_id2uid and item_id in item_id2iid:
        # if user_id in user_id2uid and item_id in item_id2iid:
            rating = 5 if rating >= 5 else rating
            all_ratings.append((user_id2uid[user_id], item_id2iid[item_id], rating))

    all_ratings = list(set(all_ratings))
    ## package

    npart = lambda l,n: ceil(len(l)/10*n)
    traindata = all_ratings[:npart(all_ratings, 6)]
    validdata = all_ratings[npart(all_ratings, 6): npart(all_ratings, 8)]
    testdata = all_ratings[npart(all_ratings, 8):]

    htry_u_lists, htry_ur_lists, htry_v_lists, htry_vr_lists = map(defaultdict, [list]*4)
    for uid, iid, rating in traindata+validdata:
        htry_u_lists[uid].append(iid)
        htry_ur_lists[uid].append(rating)
        htry_v_lists[iid].append(uid)
        htry_vr_lists[iid].append(rating)

    item_adj_lists = defaultdict(list)
    sorted_list = [(k, len(v)) for k, v in htry_v_lists.items()]
    sorted_list.sort(key = lambda x: x[1])
    sorted_list = [x[0] for x in sorted_list]
    traversed = set()
    for iid in tqdm(sorted_list, ncols=80, desc='building item adj'):
        traversed |= set([iid])
        two_hops = set()
        for uid in htry_v_lists[iid]:
            two_hops |= set(htry_u_lists[uid])
        for nid in two_hops:
            if nid in traversed or nid == iid:
                continue
            if len(set(htry_v_lists[iid]) & set(htry_v_lists[nid]))/len(set(htry_v_lists[iid]) | set(htry_v_lists[nid])) > 0.5:
                    item_adj_lists[iid].append(nid)
                    item_adj_lists[nid].append(iid)

    assert set(user_id2uid.values()) == set(range(len(user_id2uid)))
    assert set(item_id2iid.values()) == set(range(len(item_id2iid)))

    # 
    user_degrees = [len(social_adj_lists[u]) for u in range(len(user_id2uid))]
    high_degree_u_list = [i for i, d in enumerate(user_degrees) if d > args.max_social_degree]
    for i, u in enumerate(high_degree_u_list):
        for v in high_degree_u_list[i:]:
            if len(social_adj_lists[u]) > args.max_social_degree and len(social_adj_lists[v])> args.max_social_degree and v in social_adj_lists[u]:
                social_adj_lists[u].remove(v)
                social_adj_lists[v].remove(u)

    pickle.dump([htry_u_lists, htry_ur_lists, htry_v_lists, htry_vr_lists, traindata, validdata, testdata, social_adj_lists, item_adj_lists, user_id2uid, item_id2iid], open(args.rdir/(args.dset+'.pkl'), 'wb'))


def _readfile(picklepath):
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, traindata, validdata, testdata, social_adj_lists, item_adj_lists, user_id2uid, item_id2iid = pickle.load(open(picklepath, 'rb'))
    Nums = [len(user_id2uid), len(item_id2iid)]
    Lists = [history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, social_adj_lists, item_adj_lists]
    Datas = [traindata, validdata, testdata]
    return Nums, Lists, Datas

def check():
    for dset in ['library', 'ciao', 'epinions']:
        Nums, Lists, Datas = _readfile(Path('cache')/(dset+'.pkl'))
        print()
        print('=====')
        print(dset, Nums)
        
        # itemitemlist = Lists[5]
        # import code 
        # code.interact(local=locals())

        print('check rating distribution')
        ratings = [0]*7
        for key in Lists[1]:
            for r in Lists[1][key]:
                ratings[r]+=1
        print(ratings, sum(ratings))

        print('degree distribution: social/item/rating')

        for thelist, name in zip([Lists[1], Lists[3], Lists[4], Lists[5]], ['user_rate', 'item_rated', 'social', 'item']):


            degrees = [0]*1000
            sumd = 0
            for key in thelist:
                degrees[len(thelist[key])]+=1
                sumd += len(thelist[key])

            for i in range(999, 1, -1):
                if degrees[i]!= 0:
                    break
            print(name, i, sumd)
# 
            # if name in ['social', 'item']:
            # else:
                # print(name, i)
            # print(degrees[:100])

        # print(degrees[:-i+1])

def main():
    random.seed(0)
    parser = argparse.ArgumentParser(description='build ciao dataset')
    parser.add_argument('--dset', type=str, default='library')
    parser.add_argument('--rdir', type=str, default=Path('cache/'))
    parser.add_argument('--min_social_degree', type=int, default=15)
    parser.add_argument('--max_social_degree', type=int, default=300)
    parser.add_argument('--min_rating_degree', type=int, default=1)
    parser.add_argument('--max_rating_degree', type=int, default=100)
    parser.add_argument('--min_item_degree', type=int, default=5)
    parser.add_argument('--max_item_degree', type=int, default=150)
    args = parser.parse_args()

    print(' >>> build {} dataset'.format(args.dset))
    if args.dset == 'ciao':
        raw_rating, raw_social = read_ciao(args)
    elif args.dset == 'epinions':
        args.min_item_degree = 1
        raw_rating, raw_social = read_epinions(args)
    elif args.dset == 'library':
        raw_rating, raw_social = read_library(args)
    else:
        errmsg = "[args.dset not set]" if args.dset is None else args.dset+ " not recognized"
        raise Exception(errmsg)

    package_data(args, raw_rating, raw_social)
    check()

if __name__ == '__main__':
    main()
    # check()
