import os
import json
from pathlib import Path

def make_experiment_folder(exp_id):
    """Cria a pasta de experimento caso não exista

    Args:
        exp_id (int): Id inteiro do experimento
    """
    if not os.path.exists(f"results/experiment_{exp_id}"):
        os.makedirs(f"results/experiment_{exp_id}")
    
    if not os.path.exists(f"results/experiment_{exp_id}/model"):
        os.makedirs(f"results/experiment_{exp_id}/model")

def parse_exp_json(exp_id):
    
    filename = f'experiments/exp_{exp_id}.json'
    
    with open(filename, 'r') as json_file:
        # Load the JSON data into a Python data structure
        data = json.load(json_file)

    save_exp = f'results/experiment_{exp_id}/exp_{exp_id}.json'
    json_object = json.dumps(data, indent=4)
    
    with open(save_exp, "w") as outfile:
        outfile.write(json_object)
    
    
    return data

def check_exp_exist(exp_id):
    filename = f'experiments/exp_{exp_id}.json'
    my_file = Path(filename)
    if my_file.is_file():
        return True
    