
# Regularizing deep net models with primate IT neural data

## What is this?

This is the code used to create all models and plots in the ICLR 2023 paper [Aligning Model and Primate Late Stage Visual Representations Improves Model-to-Human Behavioral Alignment and Adversarial Robustness](https://openreview.net/forum?id=SMYdcXjJh1q)

Currently this repo is being cleaned up and modified to include pretrained models and data for training and validation.

## Example model training round to produce an IT-aligned CORnet-S:

```
python main.py -v --seed 0 --neural_loss logCKA --arch cornet_s --epochs 1200 --save_path test_round -nd sachimajajhongpublic -s All -n All -aei \
    --loss_weights 1 1 1 -mix_rate 1 -causal 1 --val_every 30
```

Note, you will need to download and situate the sachimajajhong dataset (temporarily located at https://ufile.io/dxuc9ghr) for training and the manymonkeys2 dataset (temporarily located at https://ufile.io/1v28f7vy) for validation.

More training scripts to recreate all model conditions and paper results can be found in the array_final*.sbatch files. You will need to install brainscore for these training scripts to work!
