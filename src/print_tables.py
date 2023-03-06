import os
import json
import torch
from statistics import mean
import argparse
from pathlib import Path
from collections import defaultdict
from experiments.single_player import score_stat

def read_value(blist, nop, dset, type_, method=None):
    msg = ""
    lower = lambda str: None if str is None else str.lower() 
    dset, type_, method = map(lower, [dset, type_, method])

    ckpt = Path('ckpt')
    values = []
    for model in ['consis']:
        for b in blist:
            if method is not None:
                filepath = ckpt/(str(nop)+'p')/'{}_{}_{}'.format(dset, type_, method)/'result_{}{}.json'.format(model, b)
                finalpath = ckpt/(str(nop)+'p')/'{}_{}_{}'.format(dset, type_, method)/'{}{}.json'.format(model, b)
            else:
                filepath = ckpt/(str(nop)+'p')/'{}_{}'.format(dset, type_)/'result_{}.json'.format(model)
                finalpath = ckpt/(str(nop)+'p')/'{}_{}'.format(dset, type_)/'{}{}.json'.format(model, b)
            
            if os.path.exists(filepath):
                both_results = json.load(open(filepath))
                numbers = score_stat(both_results)
                values.append(numbers['avg'])
                values.append(numbers['top3'])
            else:
                values.append(-1)
                values.append(-1)
    return values
    
def print_table(blist, dset_list):
    nop= 2
    rd = {}
    for dset_name in dset_list:
        type_method_dict = {}
        for type_name, method_name in zip(['MCA', 'CA', 'CA', 'IA', 'IA', 'IA'], ['MSOPS', 'BOPS', 'Random', 'BOPS', 'Popular', 'Random']):
            type_method_dict[(type_name, method_name)] = read_value(blist, nop, dset_name, type_name, method_name)
        type_method_dict['none'] = read_value(blist, nop, dset_name, 'none')
        type_method_dict['PGA'] = read_value(blist, nop, dset_name, 'pga', 'none')
        type_method_dict['\\textit{{S}}-attack'] = read_value(blist, nop, dset_name, 'srwa', 'none')
        type_method_dict['RevAdv'] = read_value(blist, nop, dset_name, 'rev', 'none')
        type_method_dict['Trial'] = read_value(blist, nop, dset_name, 'trial', 'none')
        rd[dset_name] = type_method_dict

    # find top
    for dset_name in dset_list:
        type_method_dict = rd[dset_name]    
        all_values = list(type_method_dict.values())
        all_tops = [max([1e-5]+[row[i] for row in all_values]) for i in range(len(all_values[0]))]
        all_seconds = [max([1e-5]+[row[i] for row in all_values if row[i]!=all_tops[i]]) for i in range(len(all_values[0]))]

        for k,values in type_method_dict.items():
            msg = ''
            for i, v in enumerate(values):
                if args.p:
                    if v == all_tops[i]:
                        msg += '& $\\pmb{{{:.4f}}}$ '.format(v)
                    elif v == all_seconds[i]:
                        msg += '& $\\underline{{{:.4f}}}$ '.format(v)
                    elif v < 0:
                        msg += '& -- '
                    else:
                        msg += '& ${:.4f}$ '.format(v)
                else:
                    msg += '\t{:.4f}'.format(v)
            type_method_dict[k] = msg

    for dset_name in dset_list:
        dset_msg = f"\t\\multirow{{11}}{{*}}{{" +dset_name+ f"}} & "
        type_name = 'IA'
        type_msg = f" \\multirow{{8}}{{*}}{{" +type_name+ f"}} & "
        print(dset_msg+type_msg)
        startmsg = '\t' if args.p else ""
        for method_name in ['None', 'Random', 'Popular', 'PGA', '\\textit{{S}}-attack', 'RevAdv', 'Trial']:
            msg = startmsg
            startmsg = "\t&&" if args.p else ""
            _method_name = method_name if args.p else method_name[:6]
            if method_name == 'None':
                msg = msg + "{} ".format(_method_name) + rd[dset_name]['none'] + "\\\\"
            elif (type_name, method_name) in rd[dset_name]:
                msg = msg + "{} ".format(_method_name) + rd[dset_name][(type_name, method_name)] + "\\\\"
            elif method_name in rd[dset_name]:
                msg = msg + "{} ".format(_method_name) + rd[dset_name][method_name] + "\\\\"
            else:
                msg = msg +  "{} ".format(_method_name) + "\\\\"
            print(msg)
        print("\t\\cline{2-13}")
        if args.p:
            print("\t& IA & BOPDS "+ rd[dset_name][('IA', 'BOPS')] + "\\\\")
            print("\t& CA & BOPDS "+ rd[dset_name][('CA', 'BOPS')] + "\\\\")
            print("\t\\cline{2-13}")
            print("\t& MCA & MSOPDS "+ rd[dset_name][('MCA', 'MSOPS')] + "\\\\")
        else:
            print("IABODS"+rd[dset_name][('IA', 'BOPS')])
            print("CABODS"+rd[dset_name][('CA', 'BOPS')])
            print("MSODS"+rd[dset_name][('MCA', 'MSOPS')])
        if dset_name != dset_list[-1]:
            print()
            print("\t\\midrule")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='base setup')
    parser.add_argument('--blist', type=str, default='1 2 3 4 5')
    parser.add_argument('-d', '--d', type=str, default='all')
    parser.add_argument('-p', action='store_true') # do plot
    args = parser.parse_args()
    args.blist = [int(i) for i in args.blist.split()]
    if len(args.d) == 3 and args.d != 'all':
        args.d = {'cia':'ciao', 'epi':'epinions', 'lib': 'library'}[args.d]
    # args.dset_list = ['Ciao', 'Epinions'] if args.d == 'all' else [args.d.capitalize()]
    args.dset_list = ['Ciao', 'Epinions', 'Library'] if args.d == 'all' else [args.d.capitalize()]

    print('TABLE PRINTING SERVICE: OPTIONS')
    print(args.blist, args.dset_list)
    print()
    print_table(args.blist, args.dset_list)
