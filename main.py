import argparse
import torch
import logging



from run import model_train, model_eval
from DatasetTypeEnum import DataSetType
from NetworksEnum import Networks

from util import make_experiment_folder, check_exp_exist

logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

torch.manual_seed(7)

def main():
    parser = argparse.ArgumentParser(description='Training on datasets using networks from scratch')
    
    parser.add_argument('-e',  '--experiment_id', type=int, default="0",  required=True, help='Id do experimento a ser executado')
    parser.add_argument('-t',  '--train', type=int, default="1",  required=True, help='Train(1) or Eval(0)')
    args = parser.parse_args()
    
    logging.basicConfig(
        filename=f"logs/exp_{args.experiment_id}.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    if check_exp_exist(args.experiment_id):
        if args.train == 1:
            logging.info(f"--------------------- Iniciando Novo Treinamento {args.experiment_id} ---------------------")
            make_experiment_folder(args.experiment_id)
            model_train(args.experiment_id)
        elif args.train == 0:
            logging.info(f"--------------------- Iniciando Teste {args.experiment_id} ---------------------")
            model_eval(args.experiment_id)
    else:
        logging.error(f"Erro! Experimento {args.experiment_id} não existe")


if __name__ == "__main__":
    main()