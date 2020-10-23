import torch

from config import cfg

from methods import medical, transductive, panet

if cfg['gpu'] and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def train_medical(resume=False):
    return medical.train_medical(device, resume=resume)


def test_medical(model=None, test_organ='liver'):
    return medical.test_medical(device, model=model, test_organ=test_organ)


def train_transductive(resume=False):
    return transductive.train_transductive(device, resume=resume)


def test_transductive(train_checkpoint_path, epochs=5):
    return transductive.test_transductive(device, train_checkpoint_path=train_checkpoint_path, epochs=epochs)


def train_panet(resume=False, dataset_name='voc'):
    return panet.train_panet(device, resume=resume, dataset_name=dataset_name)


def test_panet(device, model=None, dataset_name='voc', test_organ='liver'):
    return panet.test_panet(device, model=model, dataset_name=dataset_name, test_organ=test_organ)
