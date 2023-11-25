import params
from funcs import *
from wideresnet import *
import numpy as np
import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def epoch(args, loader, model, teacher = None, lr_schedule = None, epoch_i = None, opt=None, stop = False):
    train_loss = 0
    train_acc = 0
    train_n = 0
    i = 0
    func = tqdm if stop == False else lambda x:x
    criterion_kl = nn.KLDivLoss(reduction = "batchmean")
    alpha, T = 1.0, 1.0

    for batch in func(loader):
        X,y = batch[0].to('cpu'), batch[1].to('cpu')
        yp = model(X)

        loss = nn.CrossEntropyLoss()(yp,y)

        if opt:
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()

        train_loss += loss.item()*y.size(0)
        train_acc += (yp.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        i += 1
        if stop:
            break
        
    return train_loss / train_n, train_acc / train_n

def epoch_test(args, loader, model, stop = False):
    """Evaluation epoch over the dataset"""
    test_loss = 0; test_acc = 0; test_n = 0
    func = lambda x:x
    with torch.no_grad():
        for batch in func(loader):
            X,y = batch[0].to('cpu'), batch[1].to('cpu')
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp,y)
            test_loss += loss.item()*y.size(0)
            test_acc += (yp.max(1)[1] == y).sum().item()
            test_n += y.size(0)
            if stop:
                break
    return test_loss / test_n, test_acc / test_n        




def get_student_teacher(args):
    mode = args.mode
    deep_full = 28
    deep_half = 16

    # Here, the attack model is - Model Distillation Attack - where adversary has commplete access
    # to victim's private training data. So, we skip training the teacher model and directly proceed
    # with the training of student model.

    if mode == 'teacher':
        teacher = None

    # python train.py --batch_size 256 --mode teacher --normalize 0 --model_id teacher_unnormalized --lr_mode 2 --epochs 100 --dataset CIFAR10 --dropRate 0.3
    # python train.py --batch_size 256 --mode teacher --normalize 1 --model_id teacher_normalized --lr_mode 2 --epochs 100 --dataset CIFAR10 --dropRate 0.3
    student =  WideResNet(n_classes = args.num_classes, depth=deep_full, widen_factor=10, normalize = args.normalize, dropRate = 0.3)
    # student = nn.DataParallel(student).to('cpu')
    student.train()

    return student, teacher


def trainer(args):
    train_loader, test_loader = get_dataloaders(args.dataset, args.batch_size)

    def myprint(a):
        print(a); file.write(a); file.write("\n"); file.flush()

    file = open(f"{args.model_dir}/logs.txt", "w") 

    student, teacher = get_student_teacher(args)
    if args.opt_type == "SGD": 
        opt = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) 
    else:
        optim.Adam(student.parameters(), lr=0.1)

    lr_schedule = lambda t: np.interp([t], [0, args.epochs//2, args.epochs], [args.lr_min, args.lr_max, args.lr_min])[0]
    t_start = 0

    for t in range(t_start, args.epochs):  
        lr = lr_schedule(t)
        student.train()
        train_loss, train_acc = epoch(args, train_loader, student, teacher = teacher, lr_schedule = lr_schedule, epoch_i = t, opt = opt)
        student.eval()
        test_loss, test_acc   = epoch_test(args, test_loader, student)
        myprint(f'Epoch: {t}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}, lr: {lr:.5f}')    
        
        if (t+1)%25 == 0:
            torch.save(student.state_dict(), f"{args.model_dir}/iter_{t}.pt")

    torch.save(student.state_dict(), f"{args.model_dir}/final.pt")




if __name__ == "__main__":
    parser = params.parse_args()
    args = parser.parse_args()
    print(args.__dict__)
    model_dir = f"models/{args.dataset}"
    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
    args.model_dir = model_dir
    with open(f"{model_dir}/model_info.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)
    n_class = {"CIFAR10":10}
    args.num_classes = n_class[args.dataset]
    trainer(args)