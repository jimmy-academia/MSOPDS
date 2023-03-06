import os
import sys
import torch
import random
from pathlib import Path
from build_demographic import inject_fake_accounts, add_a_rate

def first_time(args, b=10):
    base_dir = Path('cache/baseline/rev')
    result_path = 'cbase/rev_{}.pt'.format(args.dataset[:3])

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    if not os.path.exists(result_path):
        for dataset in ['epinions', 'ciao', 'library']:
            print('checking for revadv: ', dataset)

            if not os.path.exists(base_dir/'input_{}.npy'.format(dataset)):
                Nums, Lists, Datas, target_items, target_users, competing_items, Domain_list = torch.load('cache/demographic/{}.pt'.format(dataset))

                rev_dset_dir = base_dir/dataset
                if not os.path.exists(rev_dset_dir):
                    os.makedirs(rev_dset_dir)
                with open(rev_dset_dir/'user2id.txt', 'w') as f:
                    for i in range(Nums[0]):
                        f.write(str(i)+'\n')
                with open(rev_dset_dir/'item2id.txt', 'w') as f:
                    for i in range(Nums[1]):
                        f.write(str(i)+'\n')
                with open(rev_dset_dir/'train.csv', 'w') as f:
                    f.write('uid,sid\n')
                    for u, v, r in Datas[0]:
                        if r >= 3:
                            f.write('{},{}\n'.format(u, v))
                with open(rev_dset_dir/'test.csv', 'w') as f:
                    f.write('uid,sid\n')
                    for u, v, r in Datas[1]:
                        if r >= 3:
                            f.write('{},{}\n'.format(u, v))

                print('execute', dataset)
                print('mv', rev_dset_dir, '../RevAdv/data/')
                print('in generate_attack_args.py change data_path=./data/{}'.format(dataset))
                print('in generate_attack.py change target_items=[{}]'.format(target_items[0]))
                print()
                
        print('to follow above instructions')
        print('todo: check where RevAdv store results and load')
        sys.exit(0)

    else:        
        Nums, Lists, Datas, target_items, target_users, competing_items, Domain_list = torch.load(args.demographic_path)
        # num_fake_users = 100
        num_fake_users = int(b*Nums[0]*0.002)
        # num_fake_users = int(b*10)
        iNums, iLists, iDatas, fake_users, considered_items = inject_fake_accounts(Nums, Lists, Datas, num_fake_users)

        result = torch.load(result_path)
        for i, u in enumerate(fake_users):
            add_a_rate(Lists, Datas, int(u), target_items[0], 5)

            filler_rates = [x for x in result if x[0]== i]
            for __, v, r in filler_rates[:100]:
                rate = 2*(r+1)+1
                rate = max(1, min(5, rate))
                rate = int(rate)
                add_a_rate(Lists, Datas, int(u), v, rate)

        return iNums, Lists, Datas

def loadRevAdv(args, b, num_fake_users):
    result_path = 'cbase/rev_{}.pt'.format(args.dataset[:3])
    Nums, Lists, Datas, target_items, target_users, competing_items, Domain_list = torch.load(args.demographic_path)
    Nums, iLists, iDatas, fake_users, considered_items = inject_fake_accounts(Nums, Lists, Datas, num_fake_users)
    
    result = torch.load(result_path)
    for i, u in enumerate(fake_users):
        add_a_rate(Lists, Datas, int(u), target_items[0], 5)

    all_users = []
    for u, v in zip(*result):
        all_users.append(u)
    all_users = list(set(all_users))

    for u, v in zip(*result):
        v = int(v)
        if u in all_users[:num_fake_users] and v != target_items[0] and v in range(Nums[1]):
            rate = random.choices([1, 2, 3,4,5], weights=[0,1,3,4,3])[0] 
            add_a_rate(Lists, Datas, fake_users[all_users.index(u)], v, rate)

    return Nums, Lists, Datas
