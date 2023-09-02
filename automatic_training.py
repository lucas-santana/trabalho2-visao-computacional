import sys
import subprocess

from util import check_exp_exist


experiments_id = [9,11,12,13,14,15,16,17,18,19,20]

script_name = open("main.py")
script = script_name.read()


for exp_id in experiments_id:
    subprocess.run(["python", "main.py", "-e", f"{exp_id}", "-t", "1"])