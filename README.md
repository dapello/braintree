
# Regularizing deep net models with primate IT neural data

## What is this?

This is the code used to create all models and plots in the anonymous neurips submission: "Aligning Model and Primate Late Stage Visual Representations Improves Model-to-Human Behavioral Alignment and Adversarial Robustness"

## Example model training round to produce an IT-aligned CORnet-S:

```
python main.py -v --seed 0 --neural_loss logCKA --arch cornet_s --epochs 1200 --save_path test_round -nd sachimajajhongpublic -s All -n All -aei \
    --loss_weights 1 1 1 -mix_rate 1 -causal 1 --val_every 30
```

Note, you will need to download and situate the sachimajajhong dataset (temporarily located at https://ufile.io/guf7fhjn) for training and the manymonkeys2 dataset (temporarily located at https://ufile.io/3p1we53t) for validation.

More training scripts to recreate all model conditions and paper results can be found in the array_final*.sbatch files. You will need to install brainscore for these training scripts to work!

This code will be updated with further documented in the near future.
