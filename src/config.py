import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action='store_true', default=True, help='Use gpu')
parser.add_argument('--train_iters', type=int, default=200, help='Train Iterations')
parser.add_argument('--test_iters', type=int, default=50, help='Test Iterations')
parser.add_argument('--nshot', type=int, default=5, help='N Shots')
parser.add_argument('--nquery', type=int, default=2, help='N Queries')
parser.add_argument('--seed', type=int, default=42, help='Seed')
args = parser.parse_args(args=[])

random.seed(args.seed)

cfg = {
    'gpu': args.gpu,
    'nshot': args.nshot,
    'nquery': args.nquery,
    'panet': {
        'save_period': 50,
        'train_iterations': args.train_iters,
        'test_iterations': args.test_iters,
        'use_pretrained': False,
        'backbone': 'unet',
        'vgg_inp_size': (417, 417),
        'unet_inp_size': (416, 416),
        'lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_milestones': [10000, 20000, 30000],
        'align_loss_scalar': 1
    },
    'transductive': {
        'unet_inp_size': (512, 256),
        'lr': 0.001,
        'backbone': 'unet',
        'train_iterations': args.train_iters,
        'test_iterations': args.test_iters,
        'batch_size': 20,
        'epochs': 20,
        'pretrain_epochs': 25,
    },
    'self_sup': {
        'optim_lr': 5e-5,
        'num_epochs': 800000,
        'batch_size': 8,
    },
    'voc': {
        'model_name': 'voc.pth',
        'root': '../data/',
        'channels': 3
    },
    'ircadb': {
        'model_name': '3dircadb.pth',
        'root': '../data/3Dircadb1/',
        'channels': 1
    },
    'visceral': {
        'model_pretrained_name': 'visceral_pretrained.ckpt',
        'path': '../data/visceral.h5',
        'valid_seg': '../data/valid_seg.json',
        'silver_path': '../data/silver_visceral.h5'
    },
    'base_models_path': '/content/gdrive/My Drive/train_models'
}
