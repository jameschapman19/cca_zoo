#$ -l tmem=8G # ram
#$ -l h_rt=24:0:0 # time
#$ -S /bin/bash
#$ -j y
#$ -N ccagame  # name of the log file
#$ -l gpu=true # for gpu
#$ -wd /cluster/project0/DREP/alawryag/ccagame
hostname
date
source /home/alawryag/sourcefile # source
source /home/alawryag/venvs/ccagame/bin/activate
#wandb login --relogin eb5e7bd0e633baad0b8103da4d1f5e400d5c29ce
#wandb agent blah
done