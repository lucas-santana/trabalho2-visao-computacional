import sys
import subprocess

from util import check_exp_exist


experiments_id = [51,52,53,54,55]

script_name = open("main.py")
script = script_name.read()


for exp_id in experiments_id:
    subprocess.run(["python", "main.py", "-e", f"{exp_id}", "-t", "1"])