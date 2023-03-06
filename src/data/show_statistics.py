import os
import pickle
from functools import reduce
from operator import add
from collections import defaultdict

if __name__ == '__main__':
    print()
    print('        name, #user, #items, #ratings,      #links')
    for dataset, name in zip(['library', 'ciao', 'epinions'], ['L.T.', 'Ciao', 'Epin.']):
    # for dataset, name in zip([], []):
        path = 'cache/{}.pkl'.format(dataset)
        if not os.path.exists(path):
            print(dataset, 'not yet prepared')
            continue
        history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, traindata, validdata, testdata, social_adj_lists, item_adj_lists, user_id2uid, item_id2iid = pickle.load(open(path, 'rb'))
        num_users = len(user_id2uid)
        num_items = len(item_id2iid)
        num_ratingsdata = len(traindata + validdata + testdata)
        rating_density =  num_ratingsdata*100/num_users/num_items
        num_connections = reduce(add, map(len, social_adj_lists.values()))//2
        connection_density =  num_connections*100/num_users/(num_users-1)
        to3f = lambda x: '({:.3f}\\%)'.format(x)
        print('        '+name, '&', num_users, '&', num_items, '&', num_ratingsdata, to3f(rating_density), '&', num_connections, to3f(connection_density), '\\\\')
    print()

    