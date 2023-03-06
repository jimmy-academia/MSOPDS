import pytz
from datetime import datetime
from experiments import single_player_exp, multi_player_exp
from utils import argument_setup

def run_experiment():
    args = argument_setup()
    print('start time', datetime.now(pytz.timezone('Asia/Taipei')).strftime('%Y-%m-%d %H:%M'))
    if args.nop == 1:
        single_player_exp(args)
    else:
        multi_player_exp(args)
    print('done time', datetime.now(pytz.timezone('Asia/Taipei')).strftime('%Y-%m-%d %H:%M'))

if __name__ == '__main__':
    run_experiment()
