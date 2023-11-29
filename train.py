import params
from funcs import *
from cnn import *
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


def get_student_teacher(args):
    mode = args.mode

    # If you're training a teacher(victim) model, then the model named 'student', trained below is actually your victim model. In this case, teacher = None
    # If you're training a student(adversary) model, then you must've already trained the victim model and it should be placed in /models/dataset/teacher/ 
    # as final.pt

    # Executed for teacher(victim) model
    if mode == 'teacher':
        teacher = None # Student becomes teacher

    # Executed for student(adversary) model
    else:
        teacher = CNN(layer_num=8)
        teacher = nn.DataParallel(teacher).to(args.device)
        path = f"models/{args.dataset}/teacher/final"
        teacher = load(teacher,path)
        teacher.eval()

    # Executed for student(adversary) model
    if mode in ["distillation", "independent"]:
        # python train.py --batch_size 1000 --mode distillation --epochs 4 --dataset MNIST
        student =  CNN(layer_num=4)
        student = nn.DataParallel(student).to(args.device)
        student.train()

    # Executed for teacher(victim) model
    else:
        # python train.py --batch_size 1000 --mode teacher --epochs 10 --dataset MNIST
        student = CNN(layer_num=8)
        student = nn.DataParallel(student).to(args.device)
        student.train()

    return student, teacher


def trainer(args):
    train_loader, test_loader = get_dataloaders(args.dataset, args.batch_size, args.student_or_teacher)
    if args.mode == "independent":
        train_loader, test_loader = test_loader, train_loader

    def myprint(a):
        print(a); file.write(a); file.write("\n"); file.flush()

    file = open(f"{args.model_dir}/logs.txt", "w") 

    student, teacher = get_student_teacher(args)
    if args.opt_type == "SGD": 
        opt = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) 
    else:
        optim.Adam(student.parameters(), lr=0.1)

    lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], [args.lr_min, args.lr_max, args.lr_max/10, args.lr_min])[0]
    t_start = 0

    for t in range(t_start, args.epochs):  
        lr = lr_schedule(t)
        student.train()
        train_loss, train_acc = epoch(args, train_loader, student, teacher = teacher, lr_schedule = lr_schedule, epoch_i = t, opt = opt)
        student.eval()
        test_loss, test_acc   = epoch_test(args, test_loader, student)
        myprint(f'Epoch: {t}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}, lr: {lr:.5f}')   

        if args.dataset == "MNIST":
            torch.save(student.state_dict(), f"{args.model_dir}/iter_{t}.pt")

    torch.save(student.state_dict(), f"{args.model_dir}/final.pt")

if __name__ == "__main__":
    parser = params.parse_args()
    args = parser.parse_args()
    print(args.__dict__)
    device = 'cpu'
    args.device = device

    # Check if a teacher(victim) model is trained or student(adversary) model is trained, and change directory structure accordingly
    if args.student_or_teacher == 'teacher':
        model_dir = f"models/{args.dataset}/{args.mode}"
    else:
        model_dir = f"models/{args.dataset}/{args.mode}"

    print(model_dir)
    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
    args.model_dir = model_dir

    with open(f"{model_dir}/model_info.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)
    
    # If you're using any other dataset, please add the number of classes to the following n_class dictionary
    # and add it to the choices parameter in params.py
    n_class = {"MNIST":10}
    args.num_classes = n_class[args.dataset]
    trainer(args)