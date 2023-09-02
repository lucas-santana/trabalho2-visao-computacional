# trabalho2-visao-computacional

# Criação do ambiente

CPU
-  conda create -n trab2_visao -c pytorch pytorch torchvision cpuonly numpy matplotlib scikit-learn seaborn

GPU
    conda create -n T2 python=3.11
    conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install numpy matplotlib tensorboard scikit-learn seaborn


conda remove -n T2 --all

# Verificar Instalação
    import torch
    torch.cuda.is_available()
# Execução do programa

python main.py -h # para ver a lista de parametros
python main.py -e 0 -t [0,1]
    




