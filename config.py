
import os
DEVICE='cuda'
EPOCHS=200
BATCH_SIZE=4
LR=0.01
ratio=0.5 #Various ratios could perform better for visualization
sample_num=3
MAXMIN=False
height,width = (256, 256)

ENCODER='resnet50'

WEIGHTS=None
name='256_6_RWKVdouble'

basedir=rf'./{name}/'

os.makedirs(basedir,exist_ok=True)
outptfile=basedir+f'{ENCODER}_{WEIGHTS}_{name}.pt'


loadstate=False
loadstateptfile=outptfile
def log(traintxt,ds):
    with open(traintxt,'a') as  f:
        f.write(ds)


import glob
import shutil



for file in glob.glob('./*.py'):
    os.makedirs(basedir+'code',exist_ok=True) 
    shutil.copyfile(file,basedir+'/'+'code/'+os.path.basename(file))




