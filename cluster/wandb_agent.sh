#$ -l tmem=16G # ram
#$ -l h_rt=60:0:0 # time
#$ -S /bin/bash
#$ -j y
#$ -N Testing  # name of the log file
#$ -l gpu=true # for gpu
#$ -wd /home/jchapman/projects/barlowtwins
hostname
date
source /home/rbarbano/source_file # source
source /home/jchapman/venvs/barlowtwins/bin/activate
wandb login --relogin f6437b9357053a0a34efc60cbfaa03087e8984f9
wandb agent blah
done