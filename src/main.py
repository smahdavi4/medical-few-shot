import config
from train import train_panet, train_transductive

# train_panet(dataset_name='ircadb')
train_transductive(
    resume=False, dataset_name='voc'
)
