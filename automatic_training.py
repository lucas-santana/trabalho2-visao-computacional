import sys
import subprocess

from util import check_exp_exist


experiments_id = [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]

script_name = open("main.py")
script = script_name.read()


for exp_id in experiments_id:
    subprocess.run(["python", "main.py", "-e", f"{exp_id}", "-t", "1"])