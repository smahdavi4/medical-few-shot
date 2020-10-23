import logging

import torch

try:
    from google.colab import drive
except:
    logging.warning("Google colab import error.")

is_mounted = False

def mount():
    global is_mounted
    if not is_mounted:
        drive.mount('/content/gdrive')
        is_mounted = True

def save_state(model_name, model, optimizer, scheduler, epoch):
    mount()
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'epoch': epoch
    }
    path = "/content/gdrive/My Drive/train_models/{}".format(model_name)
    torch.save(state, path)

def load_state(model_name, model, optimizer=None, scheduler=None):
    mount()
    path = "/content/gdrive/My Drive/train_models/{}".format(model_name)
    state = torch.load(path)
    model.load_state_dict(state['model'])
    if optimizer:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler:
        scheduler.load_state_dict(state['scheduler'])
    return state['epoch']
