#$ -l tmem=16G # ram
#$ -l h_rt=60:0:0 # time
#$ -S /bin/bash
#$ -j y
#$ -N Testing  # name of the log file
#$ -l gpu=true # for gpu
#$ -wd /home/jchapman/projects/ccagame
hostname
date
source /home/jchapman/jaxsourcefile # source
source /home/jchapman/venvs/blockeigengame/bin/activate
wandb login --relogin f6437b9357053a0a34efc60cbfaa03087e8984f9
wandb agent blah
done