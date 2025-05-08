import sys
from common.utils import create_dir

path = '/split/chuac/'
create_dir(path)
with open(path+'train.txt', 'w') as f:
    for i in range(1, 21):
        f.write(f"{i}\n")

with open(path+'test.txt', 'w') as f:
    for i in range(21, 31):
        f.write(f"{i}\n")

