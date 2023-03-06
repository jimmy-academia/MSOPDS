import os
import torch
from build_demographic import inject_fake_accounts, fuse_player, add_a_rate
from attacks.singleplayer_attack import BOPS, pop_rand_sel
from recsys import execute_recsys, rerun_recsys
from statistics import mean
import json

def single_player_exp(args):
    print('[EXP] single player task >> {} <<  for [{}] on ({}rec)'.format(args.task_method, args.dataset, args.model ))

    if args.task == 'none':
        control_exp(args)
    elif args.task == 'ia':
        injection_attack_exp(args)
    elif args.task in ['ca', 'car']:
        comprehensive_attack_exp(args)
    else:
        raise Exception(args.task + " not good for single player")

def score_stat(both_results):
    result, compresult = both_results
    if type(result) == list:
        result, compresult = list(map(torch.FloatTensor, [result, compresult]))
    avg = float(result.mean())
    score = {'avg': avg}
    ranks = [list(x).index(0) for x in torch.argsort(torch.cat((result.T, compresult.T)).T, 1, True)]
    score['top3'] = sum([1 for x in ranks if x <3])/len(ranks)
    return score

def control_exp(args):
    Nums, Lists, Datas, target_items, target_users, competing_items, Domain_list = torch.load(args.demographic_path)
    target_items = [target_items[0]]
    control_results = score_stat(execute_recsys(args, Nums, Lists, Datas, target_users, target_items, competing_items))
    print(control_results)


def injection_attack_exp(args):
    Nums, Lists, Datas, target_items, target_users, competing_items, Domain_list = torch.load(args.demographic_path)
    target_items = [target_items[0]]
    for b in args.blist: #[1,3,5]
        poisonpath = args.task_dir/'poison{}.pt'.format(b)
        if os.path.exists(poisonpath):
            continue
        num_fake_users = int(Nums[0]*b*0.001)
        iNums, iLists, iDatas, fake_users, considered_items = inject_fake_accounts(Nums, Lists, Datas, num_fake_users, target_items)
        for u in fake_users:
            add_a_rate(iLists, iDatas, int(u), target_items[0], 5)
    
        fdomain = [fake_users, considered_items]
        domain = [fdomain, None, None]
        if args.method == 'bops':
            inj = BOPS(args, domain, iNums, iLists, iDatas, target_users, target_items, competing_items, args.num_proxy_items)
        elif args.method == 'popular':
            popularities = [torch.FloatTensor(range(len(considered_items))), None, None]
            inj = pop_rand_sel(domain, args.num_proxy_items, popularities)
        elif args.method == 'random':
            inj = pop_rand_sel(domain, args.num_proxy_items)

        torch.save(inj, poisonpath)

def comprehensive_attack_exp(args):
    for b in args.blist:
        Nums, Lists, Datas, target_items, target_users, competing_items, Domain_list = torch.load(args.demographic_path)
        target_items = [target_items[0]]
        domain = Domain_list[0]
        num_fake_users = int(b*Nums[0]*0.001)
        for fid in domain[1][1][:num_fake_users]:
            add_a_rate(Lists, Datas, fid, target_items[0], 5)

        poisonpath = args.task_dir/'poison{}.pt'.format(b)
        if os.path.exists(poisonpath):
            print(poisonpath, 'exists and skipped')
            continue
        budget = b*3
        if args.method == 'bops':
            mLists, mDatas = fuse_player(domain, None, Lists, Datas)
            sel = BOPS(args, domain, Nums, mLists, mDatas, target_users, target_items, competing_items, budget)
        elif args.method == 'popular':
            sel = pop_rand_sel(domain, budget)
        elif args.method == 'random':
            sel = pop_rand_sel(domain, budget)

        torch.save(sel, poisonpath)
