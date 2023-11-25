import argparse
import yaml
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Adversarial Training')
    # Basics
    parser.add_argument("--config_file", help="Configuration file containing parameters", type=str)
    parser.add_argument("--dataset", help="CIFAR10/OTHERS", type=str, default = "CIFAR10",  choices=["CIFAR10"])
    parser.add_argument("--batch_size", help = "Batch Size for Train Set (Default = 256)", type = int, default = 256)
    parser.add_argument("--normalize", help = "Normalize training data inside the model", type = int, default = 1, choices = [0,1])
    parser.add_argument("--epochs", help = "Number of Epochs", type = int, default = 6)

    # Threat models    
    parser.add_argument("--mode", help = "Threat Models", type = str, default = 'teacher', choices = ['teacher'])

    # LR
    parser.add_argument("--opt_type", help = "Optimizer", type = str, default = "SGD")
    parser.add_argument("--lr_max", help = "Max LR", type = float, default = 0.1)
    parser.add_argument("--lr_min", help = "Min LR", type = float, default = 0.)
    parser.add_argument("--seed", help = "Seed", type = int, default = 0)

    #Lp Norm Dependent
    parser.add_argument("--distance", help="Type of Adversarial Perturbation", type=str)#, choices = ["linf", "l1", "l2", "vanilla"])
    parser.add_argument("--randomize", help = "For the individual attacks", type = int, default = 0, choices = [0,1,2])
    parser.add_argument("--alpha_l_1", help = "Step Size for L1 attacks", type = float, default = 1.0)
    parser.add_argument("--alpha_l_2", help = "Step Size for L2 attacks", type = float, default = 0.01)
    parser.add_argument("--alpha_l_inf", help = "Step Size for Linf attacks", type = float, default = 0.001)
    parser.add_argument("--num_iter", help = "PGD iterations", type = int, default = 5)
    parser.add_argument("--k", help = "For L1 attack", type = int, default = 100)

    # TEST
    parser.add_argument("--regressor_embed", help = "Victim Embeddings for training regressor", type = int, default = 0, choices = [0,1])
    return parser

