import sys
import subprocess

from util import check_exp_exist

experiments_id = [1,2,3,4,9,10,11,15,16,17]

script_name = open("main.py")
script = script_name.read()


for exp_id in experiments_id:
    subprocess.run(["python", "main.py", "-e", f"{exp_id}", "-t", "1"])