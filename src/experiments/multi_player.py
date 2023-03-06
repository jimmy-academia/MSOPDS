import os
import time
import json
import torch
from build_demographic import inject_fake_accounts, fuse_player, fuse_all_players, add_a_rate
from attacks.singleplayer_attack import BOPS
from attacks.multiplayer_attack import MSOPS
from attacks.new_mcaa import MSOPS_acc

from experiments.single_player import score_stat
from recsys import execute_recsys

from attacks.itsrwa import doSRWAattack
from attacks.pga import loadPGAattack
from attacks.revisit import loadRevAdv
from attacks.trial import loadTrial

def multi_player_exp(args): 
    print('[EXP] {} player task >> {} <<  for [{}] on ({}rec)'.format(args.nop, args.task_method, args.dataset, args.model))
    print('against opponents with car-bops')
    
    for b in args.blist:
        Nums, Lists, Datas, target_items, target_users, competing_items, Domain_list = torch.load(args.demographic_path)
        num_fake_users = int(b*Nums[0]*0.001)
        print('attacker budget', num_fake_users)
        if args.task in ['ia', 'car', 'ca']:
            # load single_player attacker poison: 
            single_poison = args.check_dir /'1p'/'_'.join([args.dataset, args.task_method])/'poison{}.pt'.format(b)
            sel = torch.load(single_poison, map_location=args.device)
            if args.task == 'ia':
                Nums, Lists, Datas, fake_users, considered_items = inject_fake_accounts(Nums, Lists, Datas, num_fake_users, target_items)
                for u in fake_users:
                    add_a_rate(Lists, Datas, int(u), target_items[0], 5)
    
                fdomain = [fake_users, considered_items]
                # fdomain = [fake_users, [target_items[0]]+considered_items]
                domain = [fdomain, None, None]
            elif args.task == 'ca':
                domain = Domain_list[0]
                for fid in domain[1][1][:num_fake_users]:
                    add_a_rate(Lists, Datas, fid, target_items[0], 5)
            Lists, Datas = fuse_player(domain, sel, Lists, Datas)

        elif args.task in ['mca', 'mcar']:
            Lists, Datas = multiplayer_comprehensive_attack_exp(args, b, num_fake_users)
        elif args.task == 'mcaa':
            Lists, Datas, timecost = mcaa(args, b, num_fake_users)
        elif args.task == 'srwa':
            Nums, Lists, Datas = doSRWAattack(args, b, num_fake_users)
        elif args.task == 'pga':
            Nums, Lists, Datas =  loadPGAattack(args, b, num_fake_users)
        elif args.task == 'rev':
            Nums, Lists, Datas = loadRevAdv(args, b, num_fake_users)
        elif args.task == 'trial':
            Nums, Lists, Datas = loadTrial(args, b, num_fake_users)

        #### opponent ####
        #### opponent ####

        if args.task != 'none':
            result_path = args.task_dir/('result_'+args.model+str(b)+'.json')
            modelpath = args.task_dir/'model_{}{}.pt'.format(args.model, b)
        else:
            result_path = args.task_dir/('result_'+args.model+'.json')
            modelpath = args.task_dir/'model_{}.pt'.format(args.model)

        if not os.path.exists(result_path):
            for op in range(1, args.nop):
                if args.task != 'mca':
                    # load from previous op!
                    opcap = '' if args.opb == args.default_opb else str(args.opb)
                    optask_dir = args.check_dir / (str(op+1)+'p'+opcap)/ '_'.join([args.dataset, args.task_method]) 
                    pb = b if args.task != 'none' else 1
                    poisonpath = optask_dir/'op{}_poison{}.pt'.format(op, pb)
                else:
                    poisonpath = args.task_dir/'op{}_poison{}.pt'.format(op, b)

                domain = Domain_list[op]
                if not os.path.exists(poisonpath):
                    mLists, mDatas = fuse_player(domain, None, Lists, Datas, neg=True)
                    print('work for opponent', op, poisonpath)
                    sel = BOPS(args, domain, Nums, mLists, mDatas, target_users, competing_items, [target_items[0]], args.opb)
                    torch.save(sel, poisonpath)
                else:
                    print('load opponent from ', poisonpath)
                    sel = torch.load(poisonpath, map_location=args.device)
                Lists, Datas = fuse_player(domain, sel, Lists, Datas, neg=True)
            
            result = execute_recsys(args, Nums, Lists, Datas, target_users, [target_items[0]], competing_items, modelpath)
            json.dump(result, open(result_path, 'w'))
        else:
            print('reloading', result_path)
            result = json.load(open(result_path))

        score = score_stat(result)
        score['timecost'] = timecost
        filepath = args.task_dir/(args.model+str(b)+'.json')
        print(score, filepath, timecost)
        json.dump(score, open(filepath, 'w'))

def mcaa(args, b, num_fake_users):
    Nums, Lists, Datas, target_items, target_users, competing_items, Domain_list = torch.load(args.demographic_path)

    poisonpath = args.task_dir/'poison{}.pt'.format(b)
    if not os.path.exists(poisonpath):
        mLists, mDatas = fuse_all_players(Domain_list[:args.nop], Lists, Datas)
        start = time.time()
        sel = MSOPS_acc(args, Domain_list, Nums, mLists, mDatas, target_users, target_items, competing_items, b)
        timecost = time.time() - start
        torch.save([sel, timecost], poisonpath)
    else:
        sel, timecost = torch.load(poisonpath, map_location=args.device)
    fLists, fDatas = fuse_player(Domain_list[0], sel, Lists, Datas)
    return fLists, fDatas, timecost
    

def multiplayer_comprehensive_attack_exp(args, b, num_fake_users):
    Nums, Lists, Datas, target_items, target_users, competing_items, Domain_list = torch.load(args.demographic_path)

    if args.method != 'onlyreal':
        for fid in Domain_list[0][1][1][:num_fake_users]:
            add_a_rate(Lists, Datas, fid, target_items[0], 5)
        
    if args.method == 'rateonly':
        print('mca domain modified for special task', args.method)
        Domain_list[0] = [Domain_list[0][0], None, None]
    elif args.method == 'rsocial':
        print('mca domain modified for special task', args.method)
        Domain_list[0] = [Domain_list[0][0], Domain_list[0][1], None]
    elif args.method == 'ritem':
        print('mca domain modified for special task', args.method)
        Domain_list[0] = [Domain_list[0][0], None, Domain_list[0][2]]
    elif args.method == 'onlyfake':
        print('mca domain modified for special task', args.method)
        Domain_list[0] = [None, Domain_list[0][1], None]
    elif args.method == 'onlyreal':
        print('mca domain modified for special task', args.method)
        Domain_list[0] = [Domain_list[0][0], None, None]

    poisonpath = args.task_dir/'poison{}.pt'.format(b)
    
    if not os.path.exists(poisonpath):
        mLists, mDatas = fuse_all_players(Domain_list[:args.nop], Lists, Datas)
        sel = MSOPS(args, Domain_list, Nums, mLists, mDatas, target_users, target_items, competing_items, b)
        torch.save(sel, poisonpath)
    else:
        sel = torch.load(poisonpath, map_location=args.device)
    fLists, fDatas = fuse_player(Domain_list[0], sel, Lists, Datas)
    return fLists, fDatas
    