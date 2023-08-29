# trabalho2-visao-computacional

# Criação do ambiente

CPU
-  conda create -n trab2_visao -c pytorch pytorch torchvision torchaudio cpuonly numpy matplotlib tensorboard

GPU

# Execução do programa

python main.py -h # para ver a lista de parametros
python main.py -n ['LENET5', 'ALEXNET', 'VGG16'] -d ['CIFAR10', 'FASHIONMNIST']
