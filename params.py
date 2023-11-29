import argparse
import yaml
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Victim and Adversary Model Training')

    # Basics
    parser.add_argument("--dataset", help="Private dataset of victim. Accessible by adversary too", type=str, default = "MNIST",  choices=["MNIST"])
    parser.add_argument("--batch_size", help = "Batch Size for Train Set (Default = 1000)", type = int, default = 1000)
    parser.add_argument("--epochs", help = "Number of Epochs (For teacher(victim) model, choose 10, for student(adversary) model, choose 5)", type = int, default = 10)
    parser.add_argument("--student_or_teacher", help = "Are you training(or generating features for) teacher(victim) model or student(adversary)", type = str, default = 'teacher', choices=['student', 'teacher'])

    # Threat models    
    parser.add_argument("--mode", help = "Threat Models. For training(or generating features for) teacher(victim) model, use 'teacher', for student(adversary) model, use one of ['distillation', 'independent]"
                        , type = str, default = 'distillation', choices = ['teacher', 'distillation', 'independent'])

    # LR
    parser.add_argument("--opt_type", help = "Optimizer", type = str, default = "SGD")
    parser.add_argument("--lr_max", help = "Max LR", type = float, default = 0.1)
    parser.add_argument("--lr_min", help = "Min LR", type = float, default = 0.)
    parser.add_argument("--seed", help = "Seed", type = int, default = 0)

    #Lp Norm Dependent
    parser.add_argument("--distance", help="Type of Adversarial Perturbation", type=str)
    parser.add_argument("--randomize", help = "For the individual attacks", type = int, default = 0, choices = [0,1,2])
    parser.add_argument("--alpha_l_1", help = "Step Size for L1 attacks", type = float, default = 1.0)
    parser.add_argument("--alpha_l_2", help = "Step Size for L2 attacks", type = float, default = 0.01)
    parser.add_argument("--alpha_l_inf", help = "Step Size for Linf attacks", type = float, default = 0.001)
    parser.add_argument("--num_iter", help = "PGD iterations", type = int, default = 10)
    parser.add_argument("--k", help = "For L1 attack", type = int, default = 100)
    parser.add_argument("--gap", help = "For L1 attack", type = float, default = 0.001)

    # TEST
    parser.add_argument("--feature_type", help = "Feature type for generation", type = str, default = 'mingd', choices = ['mingd'])
    return parser

