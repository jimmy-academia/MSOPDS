
import os
import sys
import torch
import random
import numpy as np
from pathlib import Path
from build_demographic import inject_fake_accounts, add_a_rate
from collections import defaultdict

def loadTrial(args, b, num_fake_users):
    base_dir = Path('cache/baseline/trial')
    result_path = 'cbase/tri_{}.pt'.format(args.dataset[:3])

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    if not os.path.exists(result_path):
        for dataset in ['epinions', 'ciao', 'library']:
            print('checking for trial: ', dataset)

            Nums, Lists, Datas, target_items, target_users, competing_items, Domain_list = torch.load('cache/demographic/{}.pt'.format(dataset))
            print('target_items', target_items)
            all_positive = defaultdict(list)
            with open(base_dir/'{}.train.rating'.format(dataset), 'w') as f:
                for u,v,r in Datas[0]:
                    f.write('{}\t{}\t{}\n'.format(u, v, r))
                    all_positive[u].append(v)

            one_rating = {}
            for u,v,r in Datas[1]:
                all_positive[u].append(v)
                if u not in one_rating:
                    one_rating[u] = (v,r)

            noshow = 0
            with open(base_dir/'{}.test.rating'.format(dataset), 'w') as f, open(base_dir/'{}.test.negative'.format(dataset), 'w') as fn:
                for u in range(Nums[0]):
                    if u not in one_rating:
                        v = random.choice(range(Nums[1]))
                        r = random.choice(range(1, 6))
                        noshow += 1
                    else:
                        v, r = one_rating[u]

                    f.write('{}\t{}\t{}\n'.format(u, v, r))
                
                    negs = list(set(range(Nums[1])) - set(all_positive[u]))
                    random.shuffle(negs)
                    negs = negs[:100]
                    negs = list(map(str, negs))
                    fn.write(str(u)+'\t'+'\t'.join(negs)+'\n')

            print('preprocessing done for trial: ', dataset, 'noshow:', noshow)

        print('preprocessing all done, move to trial repository:', base_dir)
        sys.exit(0)
    else:        
        Nums, Lists, Datas, target_items, target_users, competing_items, Domain_list = torch.load(args.demographic_path)
        Nums, iLists, iDatas, fake_users, considered_items = inject_fake_accounts(Nums, Lists, Datas, num_fake_users)

        result = torch.load(result_path)
        for i, u in enumerate(fake_users):
            add_a_rate(Lists, Datas, int(u), target_items[0], 5)
            fillers = [x for x in result if x[0]== i]
            for __, v in fillers[:100]:
                v = int(v)
                if v != target_items[0] and v in range(Nums[1]):
                    rate = random.choices([1, 2, 3,4,5], weights=[0,1,3,4,3])[0] 
                    add_a_rate(Lists, Datas, int(u), v, rate)

        return Nums, Lists, Datas
