import random
import torch
import higher
from tqdm import tqdm
from data.dataloading import _make_loaders
from networks.surrogate import Probabilstic_Surrogate

def BOPS(args, domain, Nums, Lists, Datas, target_users, target_items, competing_items, budget):
    train_loader, __, __ = _make_loaders(Datas, args.batch_size)
    print('BODS budget is', budget)
    selections = []
    for dom in domain:
        if dom is not None and len(dom[0])*len(dom[1]) > 0:
            sel = torch.ones(len(dom[0]), len(dom[1])) * 0.1
            sel += torch.rand_like(sel)*0.0001
            sel = sel.to(args.device)
            selections.append(sel)
        else:
            selections.append(None)
    
    pbar = tqdm(range(args.adv_rounds), ncols=88, desc='BOPS')
    for rd in pbar:   
        varsels = []
        for s, sel in enumerate(selections):
            if sel is not None:
                varsel = sel.clone()
                if s == 1:
                    varsel = varsel.T
                varsel = make_vector_selection(varsel, budget)
                if s == 1:
                    varsel = varsel.T
                varsel.requires_grad = True
                varsels.append(varsel)
            else:
                varsels.append(None)

        sur_model = Probabilstic_Surrogate(args, Nums, Lists, [domain])
        sur_optimizer = torch.optim.Adam(sur_model.parameters())
        
        with higher.innerloop_ctx(sur_model, sur_optimizer) as (fmodel, diffopt):
            for epoch in range(1, args.epochs+1):
                fmodel.train()
                dbar = tqdm(train_loader, ncols=88, leave=False, desc='ep->s: '+str(epoch))
                for data in dbar:
                    batch_nodes_u, batch_nodes_v, labels_ = list(map(lambda x:x.to(args.device), data))
                    loss = fmodel.frac_loss(batch_nodes_u, batch_nodes_v, labels_, [varsels])
                    diffopt.step(loss)
                    dbar.set_postfix(recsysloss=float(loss)/len(labels_)) 
            fmodel.eval()
            res = fmodel.grid_results(target_users, target_items, [varsels])
            cres = fmodel.grid_results(target_users, competing_items, [varsels])
            mean_loss = res.mean() - args.compete_coeff * cres.mean()
            rank_loss = torch.neg(torch.nn.SELU()(cres - res)).mean()
            loss = mean_loss + 4 * rank_loss
            loss.backward()
            for i in range(3):
                if varsels[i] is not None and varsels[i].grad is not None:
                    grad = varsels[i].grad.detach()
                    grad = torch.nn.functional.softsign(grad)
                    selections[i] += 0.1 * grad
            pbar.set_postfix(l=float(loss))
            
    varsels = []

    for s, sel in enumerate(selections):
        if sel is not None:
            if s == 1:
                sel = sel.T
            varsel = make_vector_selection(sel, budget)
            if s == 1:
                varsel = varsel.T
            varsels.append(varsel)
        else:
            varsels.append(None)

    return varsels

def pop_rand_sel(domain, budget, popularities=None):

    all_sel = []
    popularities = [None, None, None] if popularities is None else popularities
    for s, dom, popularity in zip([0,1,2], domain, popularities):
        if dom is not None:
            if s == 1:
                all_sel.append(make_vector_selection(torch.rand(len(dom[1]), len(dom[0])), budget, popularity).T)
            else:
                all_sel.append(make_vector_selection(torch.rand(len(dom[0]), len(dom[1])), budget, popularity))
        else:
            all_sel.append(None)

    return all_sel

def make_vector_selection(selection, total_budget, popularity=None):

    if total_budget == 0:
        return torch.zeros_like(selection)

    budget = total_budget if popularity is None else int(0.9*total_budget)
    popbudget = total_budget - budget

    if selection.shape[1] == 1:
        threshold_val = selection.flatten().sort(0,True)[0][budget-1]
        output = torch.where(selection>=threshold_val, torch.ones_like(selection), torch.zeros_like(selection))

        if popbudget > 0:
            popsel = (1-output)*popularity.unsqueeze(1)
            threshold_val = popsel.flatten().sort(0,True)[popbudget-1]
            output += torch.where(popsel>=threshold_val, torch.ones_like(popsel), torch.zeros_like(popsel))

    else:
        output = []
        for sel in selection:
            threshold_val = sel.sort(0,True)[0][budget-1]
            out = torch.where(sel>=threshold_val, torch.ones_like(sel), torch.zeros_like(sel))
            if popbudget > 0:
                popsel = (1-out)*popularity
                threshold_val = popsel.sort(0,True)[0][budget-1]
                out += torch.where(popsel>=threshold_val, torch.ones_like(popsel), torch.zeros_like(popsel))
            output.append(out)

        output = torch.stack(output)
    return output



