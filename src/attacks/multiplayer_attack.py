import os
import torch
import random
from tqdm import tqdm
import higher
from scipy import sparse
import scipy.sparse.linalg
import numpy as np

from data.dataloading import _make_loaders
from networks.surrogate import Probabilstic_Surrogate
from attacks.singleplayer_attack import make_vector_selection

def MSOPS(args, Domain_list, Nums, Lists, Datas, target_users, target_items, competing_items, b):
    train_loader, __, __ = _make_loaders(Datas, args.batch_size)
    budget = b*3
    opbudget = args.opb
    print('budget', budget, 'opponent budget', opbudget)
    All_selections = []
    for domain in Domain_list[:args.nop]:
        selections = []
        for dom in domain:
            if dom is not None and len(dom[0])*len(dom[1]) > 0:
                sel = torch.ones(len(dom[0]), len(dom[1])) * 0.1
                sel += torch.rand_like(sel)*0.001
                sel = sel.to(args.device)
                selections.append(sel)
            else:
                selections.append(None)
        All_selections.append(selections)

    pbar = tqdm(range(args.adv_rounds), ncols=90, desc='multi')

    for rd in pbar:
        all_sels = []
        for i, selections in enumerate(All_selections):
            varsels = []
            for s, sel in enumerate(selections):
                if sel is not None:
                    varsel = sel.clone()
                    if i == 0:  
                        if s == 1:
                            varsel = varsel.T
                        varsel = make_vector_selection(varsel, budget)
                        if s == 1:
                            varsel = varsel.T
                    else:
                        varsel = make_vector_selection(varsel, opbudget)
                    varsel.requires_grad = True
                    varsels.append(varsel)
                else:
                    varsels.append(None)
            all_sels.append(varsels)
        
        sur_model = Probabilstic_Surrogate(args, Nums, Lists, Domain_list[:args.nop])
        sur_optimizer = torch.optim.Adam(sur_model.parameters(), lr=1e-2, weight_decay=1e-5)

        with higher.innerloop_ctx(sur_model, sur_optimizer) as (fmodel, diffopt):
            for __ in tqdm(range(args.epochs), ncols=88, leave=False, desc='higher'):
                fmodel.train()
                dbar = tqdm(train_loader, ncols=88, leave=False, desc='data...')
                for data in dbar:
                    batch_nodes_u, batch_nodes_v, labels_ = list(map(lambda x:x.to(args.device), data))
                    loss = fmodel.frac_loss(batch_nodes_u, batch_nodes_v, labels_, all_sels)
                    diffopt.step(loss)
                    dbar.set_postfix(train_loss=float(loss)/len(labels_))
            
            Losses = []
            for ps in range(args.nop):
                if ps == 0:
                    res = fmodel.grid_results(target_users, [target_items[ps]], all_sels)
                    cres = fmodel.grid_results(target_users, competing_items, all_sels)
                else:
                    res = fmodel.grid_results(target_users, competing_items, all_sels)
                    otheritems = [target_items[0]]
                    cres = fmodel.grid_results(target_users, otheritems, all_sels)

                mean_loss = res.mean() - args.compete_coeff*cres.mean()
                selu_loss = torch.neg(torch.nn.SELU()(cres - res)).mean() 
                loss = mean_loss + 4*selu_loss
                Losses.append(loss)

            action_num = 3 if args.task == 'mca' else 1
            for ai in range(action_num):
                pbar.set_postfix(tag='do action {}'.format(ai))

                if All_selections[0][ai] is None:
                    if ai == 0:
                        for ps in range(1, args.nop):
                            opgrad = torch.autograd.grad(Losses[ps], all_sels[ps][ai], retain_graph=True)[0].detach()
                            All_selections[ps][ai] += 0.05 * torch.nn.functional.softsign(opgrad)
                    continue


                main_grad = torch.autograd.grad(Losses[0], all_sels[0][ai], retain_graph=True, create_graph=True)[0]
                if ai == 0:
                    for ps in range(1, args.nop):
                        pbar.set_postfix(tag='do action {} - oppo:{}'.format(ai, ps))

                        opgrad = torch.autograd.grad(Losses[ps], all_sels[ps][ai], retain_graph=True)[0].detach()
                        All_selections[ps][ai] += 0.05 * torch.nn.functional.softsign(opgrad)

                            
                        pbar.set_postfix(tag='do action {} - oppo:{}.. solve..Df_f'.format(ai, ps))
                        Df_f = torch.autograd.grad(Losses[ps], all_sels[ps][ai], retain_graph=True, create_graph=True)[0]
                        pbar.set_postfix(tag='do action {} - oppo:{}.. solve..Dl_f'.format(ai, ps))
                        Dl_f = torch.autograd.grad(Losses[0], all_sels[ps][ai], retain_graph=True, create_graph=True)[0]
                        pbar.set_postfix(tag='do action {} - oppo:{}.. solve..DDf_f'.format(ai, ps))
                        DDf_f = JacobianVectorProduct(Df_f, all_sels[ps][ai], args.device, 1)
                        pbar.set_postfix(tag='do action {} - oppo:{}.. solve..cg'.format(ai, ps))
                        w, status = sparse.linalg.cg(DDf_f, Dl_f.cpu().detach().view(-1), maxiter=5)
                        q = torch.Tensor(JacobianVectorProduct(Df_f, all_sels[0][ai], args.device)(w)).to(args.device)
                        main_grad -= q.view(main_grad.shape)

                grad = main_grad.detach()
                grad = 0.1* torch.nn.functional.softsign(grad)
                All_selections[0][ai] += grad
                gg = [float(random.choice(grad.flatten())) for __ in range(5)]
                pbar.set_postfix(l=float(Losses[0]), gg=gg)

        varsels = []
        for s, sel in enumerate(All_selections[0]):
            if sel is not None:
                varsel = sel.clone()
                if s == 1:
                    varsel = varsel.T
                varsel = make_vector_selection(varsel, budget)
                if s == 1:
                    varsel = varsel.T
                varsels.append(varsel)
            else:
                varsels.append(None)

    return varsels


class JacobianVectorProduct(sparse.linalg.LinearOperator):
    def __init__(self, grad, params, device, regularization=0):
        if isinstance(grad, (list, tuple)):
            grad = list(grad)
            for i, g in enumerate(grad):
                grad[i] = g.view(-1)
            self.grad = torch.cat(grad)
        elif isinstance(grad, torch.Tensor):
            self.grad = grad.view(-1)

        nparams = sum(p.numel() for p in params)
        self.shape = (nparams, self.grad.size(0))
        self.dtype = np.dtype(np.float32)
        self.params = params
        self.regularization = regularization
        self.device = device

    def _matvec(self, v):
        v = torch.Tensor(v)
        if self.grad.is_cuda:
            v = v.cuda(self.device)

        hv = torch.autograd.grad(self.grad, self.params, v, retain_graph=True, allow_unused=True)
        _hv = []
        for g, p in zip(hv, self.params):
            if g is None:
                g = torch.zeros_like(p)
            _hv.append(g.contiguous().view(-1))
        if self.regularization != 0:
            hv = torch.cat(_hv) + self.regularization*v
        else:
            hv = torch.cat(_hv) 
        return hv.cpu()
    
    def _matmat(self, X):
        return np.hstack([self._matvec(col) for col in X.T])
        
