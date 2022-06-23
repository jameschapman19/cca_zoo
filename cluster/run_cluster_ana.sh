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
python3 ./experiments/pls/train_ukbb.py
done