from funcs import *
import params
import sys, time, argparse, params, glob, os, json
import torch
from train import epoch_test
from attacks import *
import time
from cnn import *

def get_student_teacher(args):
    mode = args.mode

    # Here, if we are generating features for teacher(victim) model, student becomes teacher and teacher is None.
    # When student(adversary) model is trained, teacher is anyways None.
    teacher = None

    if mode in ['distillation', 'independent']:
        # python generate_features.py --batch_size 500 --mode distillation --normalize 1 --model_id distillation_normalized
        student =  CNN(layer_num=4)

    else:
        # Executed for teacher(victim) model
        # python generate_features.py --batch_size 1000 --mode teacher 
        student =  CNN(layer_num=8)
        
    return student, teacher

def get_mingd_vulnerability(args, loader, model, num_images = 1000):
    batch_size = args.batch_size
    max_iter = num_images/batch_size
    lp_dist = [[],[],[]]
    ex_skipped = 0
    for i,batch in enumerate(loader):
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
    train_loader, test_loader = get_dataloaders(args.dataset, args.batch_size, args.student_or_teacher)
    student, _ = get_student_teacher(args) 
    location = f"{args.model_dir}/final.pt"
    try:
        student = student.to(args.device)
        student.load_state_dict(torch.load(location, map_location = args.device)) 
    except:
        student = nn.DataParallel(student).to(args.device)
        student.load_state_dict(torch.load(location, map_location = args.device))

    student.eval()
    
    # _, train_acc  = epoch_test(args, train_loader, student)
    _, test_acc   = epoch_test(args, test_loader, student, stop = True)
    print(f'Model: {args.model_dir} | \t Test Acc: {test_acc:.3f}')

    test_d = get_mingd_vulnerability(args, test_loader, student)
    torch.save(test_d, f"{args.file_dir}/test_mingd_vulnerability_2.pt")

    train_d = get_mingd_vulnerability(args, train_loader, student)
    torch.save(train_d, f"{args.file_dir}/train_mingd_vulnerability_2.pt")


if __name__ == "__main__":
    parser = params.parse_args()
    args = parser.parse_args()
    print(args.__dict__)

    device = 'cpu'
    args.device = device

    # Check if a teacher(victim) model is trained or student(adversary) model is trained, and change directory structure accordingly
    if args.student_or_teacher == 'teacher':
        model_dir = f"models/{args.dataset}/teacher"
    else:
        model_dir = f"models/{args.dataset}/student"
    args.model_dir = model_dir

    # Check if generated features are for teacher(victim) model or student(adversary) model, and change directory structure accordingly
    if args.student_or_teacher == 'teacher':
        file_dir = f"files/{args.dataset}/teacher"
    else:
        file_dir = f"files/{args.dataset}/student"
    args.file_dir = file_dir
    print("File Directory:", file_dir)

    with open(f"{model_dir}/model_info.txt", "w") as f:
            json.dump(args.__dict__, f, indent=2)
    args.device = 'cpu'
    torch.manual_seed(args.seed)

    n_class = {"MNIST":10}
    args.num_classes = n_class[args.dataset]
    
    feature_extractor(args)