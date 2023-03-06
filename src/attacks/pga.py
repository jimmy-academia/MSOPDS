import os
import sys
import torch
import random
import numpy as np
from pathlib import Path
from build_demographic import inject_fake_accounts, add_a_rate

def loadPGAattack(args, b, num_fake_users):
    base_dir = Path('cache/baseline/pga')
    result_path = 'cbase/pga_{}.pt'.format(args.dataset[:3])

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    if not os.path.exists(result_path):
        if not os.path.exists(base_dir/'input_{}.npy'.format(args.dataset)):
            Nums, Lists, Datas, target_items, target_users, competing_items, Domain_list = torch.load(args.demographic_path)
            np.save(base_dir/'input_{}.npy'.format(args.dataset), Datas[0])
        print('run `python main.py ../code_adv_recsys/cache/baseline/pga/input_{}.npy save{}.npy`'.format(args.dataset, b))
        print('in ...')
        print('to produce')
        sys.exit(0)

    else:        
        Nums, Lists, Datas, target_items, target_users, competing_items, Domain_list = torch.load(args.demographic_path)
        iNums, iLists, iDatas, fake_users, considered_items = inject_fake_accounts(Nums, Lists, Datas, num_fake_users)

        result = torch.load(result_path)
        for i, u in enumerate(fake_users):
            add_a_rate(Lists, Datas, int(u), target_items[0], 5)

            filler_rates = [x for x in result if x[0]== i]
            for __, v, r in filler_rates[:100]:
                v = int(v)
                if v in range(Nums[1]) and v != target_items[0]:
                    rate = 2*(r+1)+1
                    rate = max(1, min(5, rate))
                    rate = int(rate)
                    add_a_rate(Lists, Datas, int(u), v, rate)

        return iNums, Lists, Datas

    
'''
python main.py --nop 2 --task pga --model consis --dataset epinions 
'''