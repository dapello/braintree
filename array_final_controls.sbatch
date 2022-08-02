#!/bin/bash
#SBATCH -c 20                # Number of cores
#SBATCH -t 8000              # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=40g            # Memory pool for all cores '(see also --mem-per-cpu)
#SBATCH --gres=gpu:QUADRORTX6000:1
#SBATCH --array=0-6
#SBATCH -o slurm_logs/score_models_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e slurm_logs/score_models_%A_%a.err  # File to which STDERR will be written, %j inserts jobid

### for multiple GPUS #SBATCH --array=0-2

export HOME='/om2/user/dapello/'
module load openmind/anaconda/3-4.0.0
source activate braintree-0.2

cd /om2/user/dapello/Code/proj_braintree/braintree-0.2/braintree

nvidia-smi

seed=3
date=220505
arch=cornet_s
loss=logCKA
labels=1

# Imagenet (control)
if [ $SLURM_ARRAY_TASK_ID == 0 ]; 
then
    mix=1
    sp=${date}-final-shufflecontrol-labels_${labels}-mix_${mix}
    python main.py -v --seed $seed --neural_loss $loss --arch $arch --epochs 1200 --save_path $sp -nd sachimajajhongpublic -s All -n All \
        -BS dicarlo.Rajalingham2018-i2n dicarlo.Kar2022human-i2n dicarlo.Kar2018-i2n dicarlo.Rajalingham2018subset-i2n -aei \
        --loss_weights 1 1 ${labels} -mix_rate $mix -causal 1 --val_every 30 --controls shuffle
fi

# Imagenet (control)
if [ $SLURM_ARRAY_TASK_ID == 1 ]; 
then
    mix=1
    sp=${date}-final-randomcontrol-labels_${labels}-mix_${mix}
    python main.py -v --seed $seed --neural_loss $loss --arch $arch --epochs 1200 --save_path $sp -nd sachimajajhongpublic -s All -n All \
        -BS dicarlo.Rajalingham2018-i2n dicarlo.Kar2022human-i2n dicarlo.Kar2018-i2n dicarlo.Rajalingham2018subset-i2n -aei \
        --loss_weights 1 1 ${labels} -mix_rate $mix -causal 1 --val_every 30 --controls random
fi
