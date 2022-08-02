
# Regularizing deep net models with primate IT neural data

## What is this?

An attempt to fit neural networks to IT neural recording data.

## Example model training round to produce an IT-aligned CORnet-S:

```
python main.py -v --seed 0 --neural_loss logCKA --arch cornet_s --epochs 1200 --save_path test_round -nd sachimajajhongpublic -s All -n All -aei \
    --loss_weights 1 1 1 -mix_rate 1 -causal 1 --val_every 30
```

Note, you will need to download and situate the sachimajajhong dataset and the manymonkeys2 datasets.

More training scripts to recreate all model conditions and paper results can be found in the array_final*.sbatch files. You will need to install brainscore for these training scripts to work!

This code will be updated and further documented in the near future.
