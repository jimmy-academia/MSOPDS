import os
import torch
from pathlib import Path
from data.dataloading import _readfile, _make_loaders
from networks.consisrec import GraphConsis
from utils import default_args
from tqdm import tqdm
from copy import deepcopy


def train(args, model, train_loader, valid_loader=None, verbose=True):

    pbar = tqdm(range(1, args.recsys_epochs+1), ncols=88, desc='>>> recsys training') if verbose else range(1, args.recsys_epochs+1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    best_validate = 1e10
    best_model_state_dict = endure_count = va_loss = 0
    for epoch in pbar:
        tr_loss = train_test_epoch(args, model, train_loader, epoch, 'training', optimizer)
        if valid_loader:
            va_loss = train_test_epoch(args, model, valid_loader, epoch, 'validate')
            if va_loss < best_validate:
                best_validate = va_loss
            else:
                endure_count += 1
                endure_limit = 5 if args.model == 'consis' else 1
                if endure_count >= endure_limit:
                    break
        if verbose:
            pbar.set_postfix(l=tr_loss, vl=va_loss, end=endure_count) 
    return model

def train_test_epoch(args, model, loader, epoch=None, msg=None, train_optimizer=False):
    model.train() if train_optimizer else model.eval()
    message = '>>> epoch {} {}'.format(epoch, msg) if epoch and msg else 'recsys...'
    total_loss = sample_count = count = 0
    dbar = tqdm(loader, ncols=88, leave=False, desc=message)
    for data in dbar:
        batch_u, batch_v, labels = data
        batch_u = batch_u.to(args.device)
        batch_v = batch_v.to(args.device)
        labels = labels.to(args.device)
        loss = model.loss(batch_u, batch_v, labels)
        if train_optimizer:
            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()
        total_loss += float(loss)
        sample_count += len(labels)
        count += 1
        avgloss = total_loss / sample_count if args.model == 'consis' else total_loss / count
        dbar.set_postfix(loss=avgloss)
    return avgloss

def execute_recsys(args, Nums, Lists, Datas, target_users, target_items, competing_items, model_path=None):
    train_loader, valid_loader, __ = _make_loaders(Datas, 256)
    model = GraphConsis(args, Nums, deepcopy(Lists)).to(args.device)
    model = train(args, model, train_loader, valid_loader)
    model.eval()
    result = model.grid_results(target_users, target_items)
    compresult = model.grid_results(target_users, competing_items)
    if model_path is not None:
        torch.save(model.state_dict(), model_path)
    return (result.tolist(), compresult.tolist())

def rerun_recsys(args, Nums, Lists, target_users, target_items, competing_items):
    model = GraphConsis(args, Nums, deepcopy(Lists)).to(args.device)
    model.load_state_dict(torch.load(args.cache_dir/'model_weights'/(args.dataset+'_'+args.model+'.pth')))
    result = model.grid_results(target_users, target_items)
    compresult = model.grid_results(target_users, competing_items)
    return (result, compresult)


if __name__ == '__main__':
    prepare_models()
    