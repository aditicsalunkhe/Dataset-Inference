from funcs import *
import params
import sys, time, argparse, params, glob, os, json
import torch
from wideresnet import *
from train import epoch_test
from attacks import *
import time

def get_student_teacher(args):
    mode = args.mode
    deep_full = 28
    deep_half = 16

    # Here, the attack model is - Model Distillation Attack - where adversary has commplete access
    # to victim's private training data. So, we skip training the teacher model and directly proceed
    # with the training of student model.

    if mode == 'teacher':
        teacher = None
        # python generate_features.py --feature_type rand --dataset SVHN --batch_size 500 --mode teacher --normalize 1 --model_id teacher_normalized
        # python generate_features.py --batch_size 500 --mode teacher --normalize 0 --model_id teacher_unnormalized --dataset CIFAR10
        # python generate_features.py --batch_size 500 --mode teacher --normalize 1 --model_id teacher_normalized --dataset CIFAR10
        student =  WideResNet(n_classes = args.num_classes, depth=deep_full, widen_factor=10, normalize = args.normalize, dropRate = 0.3)
        
        #Alternate student models: [lr_max = 0.01, epochs = 100], [preactresnet], [dropRate]
    return student, teacher

def get_mingd_vulnerability(args, loader, model, num_images = 1000):
    batch_size = args.batch_size
    max_iter = num_images/batch_size
    lp_dist = [[],[],[]]
    ex_skipped = 0
    for i,batch in enumerate(loader):
        if args.regressor_embed == 1: ##We need an extra set of `distinct images for training the confidence regressor
            if(ex_skipped < num_images):
                y = batch[1]
                ex_skipped += y.shape[0]
                continue
        for j,distance in enumerate(["linf", "l2", "l1"]):
            temp_list = []
            for target_i in range(args.num_classes):
                X,y = batch[0].to('cpu'), batch[1].to('cpu') 
                args.distance = distance
                # args.lamb = 0.0001
                delta = mingd(model, X, y, args, target = y*0 + target_i)
                yp = model(X+delta) 
                distance_dict = {"linf": norms_linf_squeezed, "l1": norms_l1_squeezed, "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta)
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            # temp_dist = [batch_size, num_classes)]
            temp_dist = torch.cat(temp_list, dim = 1)
            lp_dist[j].append(temp_dist) 
        if i+1>=max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    lp_d = [torch.cat(lp_dist[i], dim = 0).unsqueeze(-1) for i in range(3)]    
    # full_d = [num_images, num_classes, num_attacks]
    full_d = torch.cat(lp_d, dim = -1); print(full_d.shape)
        
    return full_d

def feature_extractor(args):
    train_loader, test_loader = get_dataloaders(args.dataset, args.batch_size)
    student, _ = get_student_teacher(args) #teacher is not needed
    location = f"{args.model_dir}/final.pt"
    try:
        student = student.to(args.device)
        student.load_state_dict(torch.load(location, map_location = args.device)) 
    except:
        # student = nn.DataParallel(student).to(args.device)
        student.load_state_dict(torch.load(location, map_location = args.device))

    student.eval()
    
    # _, train_acc  = epoch_test(args, train_loader, student)
    _, test_acc   = epoch_test(args, test_loader, student, stop = True)
    print(f'Model: {args.model_dir} | \t Test Acc: {test_acc:.3f}')

    test_d = get_mingd_vulnerability(args, test_loader, student, 750)
    torch.save(test_d, f"{args.file_dir}/test_{args.feature_type}_vulnerability_2.pt")

    train_d = func(args, train_loader, student)
    torch.save(train_d, f"{args.file_dir}/train_{args.feature_type}_vulnerability_2.pt")


if __name__ == "__main__":
    parser = params.parse_args()
    args = parser.parse_args()
    print(args.__dict__)
    model_dir = f"models/{args.dataset}"
    print("Model Directory:", model_dir)
    args.model_dir = model_dir
    file_dir = f"files/{args.dataset}"
    print("File Directory:", file_dir)
    args.file_dir = file_dir
    if(not os.path.exists(file_dir)):
        os.makedirs(file_dir)
    with open(f"{model_dir}/model_info.txt", "w") as f:
            json.dump(args.__dict__, f, indent=2)
    args.device = 'cpu'
    torch.manual_seed(args.seed)

    n_class = {"CIFAR10":10}
    args.num_classes = n_class[args.dataset]
    
    feature_extractor(args)