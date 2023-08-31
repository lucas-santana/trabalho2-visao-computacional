import argparse
import torch

from torch.utils.tensorboard import SummaryWriter

from run import model_train
from DatasetTypeEnum import DataSetType
from NetworksEnum import Networks

from util import make_experiment_folder

writer = SummaryWriter()

torch.manual_seed(7)

def main():
    parser = argparse.ArgumentParser(description='Training on datasets using networks from scratch')
    
    parser.add_argument('-e',  '--experiment_id', type=int, default="0",  required=True, help='Id do experimento a ser executado')

    args = parser.parse_args()

    make_experiment_folder(args.experiment_id)

    model_train(args.experiment_id)
    
    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()