import sys
import subprocess

from util import check_exp_exist


experiments_id = [10,16,17]

script_name = open("main.py")
script = script_name.read()


# for exp_id in experiments_id:
for exp_id in range(1,92):
    subprocess.run(["python", "main.py", "-e", f"{exp_id}", "-t", "0"])