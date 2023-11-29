import time
import torch

def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

def norms_l2(Z):
    return norms(Z)

def norms_l2_squeezed(Z):
    return norms(Z).squeeze(1).squeeze(1).squeeze(1)

def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None]

def norms_l1_squeezed(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None].squeeze(1).squeeze(1).squeeze(1)

def norms_l0(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float()

def norms_l0_squeezed(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float().squeeze(1).squeeze(1).squeeze(1)

def norms_linf(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

def norms_linf_squeezed(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0]
def loss_mingd(preds, target):
    loss =  (preds.max(dim = 1)[0] - preds[torch.arange(preds.shape[0]),target]).mean()
    assert(loss >= 0)
    return loss

def kthlargest(tensor, k, dim=-1):
    val, idx = tensor.topk(k, dim = dim)
    return val[:,:,-1], idx[:,:,-1]

def l1_dir_topk(grad, delta, X, gap, k = 10) :
    #Check which all directions can still be increased such that
    #they haven't been clipped already and have scope of increasing
    # ipdb.set_trace()
    X_curr = X + delta
    batch_size = X.shape[0]
    channels = X.shape[1]
    pix = X.shape[2]
    # print (batch_size)
    neg1 = (grad < 0)*(X_curr <= gap)
    neg2 = (grad > 0)*(X_curr >= 1-gap)
    neg3 = X_curr <= 0
    neg4 = X_curr >= 1
    neg = neg1 + neg2 + neg3 + neg4
    u = neg.view(batch_size,1,-1)
    grad_check = grad.view(batch_size,1,-1)
    grad_check[u] = 0

    kval = kthlargest(grad_check.abs().float(), k, dim = 2)[0].unsqueeze(1)
    k_hot = (grad_check.abs() >= kval).float() * grad_check.sign()
    return k_hot.view(batch_size, channels, pix, pix)

def mingd(model, X, y, args, target):
    start = time.time()
    is_training = model.training
    model.eval()                    # Need to freeze the batch norm and dropouts
    criterion = loss_mingd
    norm_map = {"l1":norms_l1_squeezed, "l2":norms_l2_squeezed, "linf":norms_linf_squeezed}
    alpha_map = {"l1":args.alpha_l_1/args.k, "l2":args.alpha_l_2, "linf":args.alpha_l_inf}
    alpha = float(alpha_map[args.distance])
    delta = torch.zeros_like(X, requires_grad=False)    
    loss = 0
    for t in range(args.num_iter):        
        if t>0: 
            preds = model(X_r+delta_r)
            new_remaining = (preds.max(1)[1] != target[remaining])
            remaining_temp = remaining.clone()
            remaining[remaining.clone()] = new_remaining
        else: 
            preds = model(X+delta)
            remaining = (preds.max(1)[1] != target)
            
        if remaining.sum() == 0: break

        X_r = X[remaining]; delta_r = delta[remaining]
        delta_r.requires_grad = True
        preds = model(X_r + delta_r)
        loss = -1* loss_mingd(preds, target[remaining])
        print(t, loss, remaining.sum().item())
        loss.backward()
        grads = delta_r.grad.detach()
        if args.distance == "linf":
            delta_r.data += alpha * grads.sign()
        elif args.distance == "l2":
            delta_r.data += alpha * (grads / norms_l2(grads + 1e-12))
        elif args.distance == "l1":
            delta_r.data += alpha * l1_dir_topk(grads, delta_r.data, X_r, args.gap, args.k)
        delta_r.data = torch.min(torch.max(delta_r.detach(), -X_r), 1-X_r) # clip X+delta_r[remaining] to [0,1]
        delta_r.grad.zero_()
        delta[remaining] = delta_r.detach()
        
    print(f"Number of steps = {t+1} | Failed to convert = {(model(X+delta).max(1)[1]!=target).sum().item()} | Time taken = {time.time() - start}")
    if is_training:
        model.train()    
    return delta