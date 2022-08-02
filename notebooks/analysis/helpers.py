import h5py as h5
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# computed in ceilings notebook with split half reliability of CKA
ceilings = {
    'CKA_magneto.var6' : 0.6000851988792419,
    'CKA_nano.var6' : 0.8338761478662491,
    'CKA_nano.coco' : 0.8167327344417572,
    'CKA_bento.coco' : 0.9570454135537148
}

def recover_time_bin_start(file_name):
    # pretty gross function to recover the time_bin_start from the file name.
    file_name = file_name.replace('time_-', 'time_m')
    time = [piece for piece in file_name.split('-') if 'time_' in piece][0]
    time = int(time.split('_')[-1].replace('m','-'))
    return time

def catch(filepath, target, ind=1, verbose=False):
    filepath_ = filepath.replace('/','-')
    parts = filepath_.split('-')
    match = [part for part in parts if target in part]
    if len(match) == 1:
        return match[0].split('_')[ind]
    elif len(match) > 1:
        if verbose:
            print('catch() found multiple matches, returning first.')
        return match[0].split('_')[ind]
    else:
        if verbose:
            print('target {} not found in filepath {}'.format(target,filepath))
        return None

def save(path, data):
    f = h5.File(path, 'w')
    f.create_dataset('obj_arr', data=data)
    f.close()    
    print('Saved ', path)

def filter_neuroids(assembly, threshold):
    from brainscore.metrics.ceiling import InternalConsistency
    from brainscore.metrics.transformations import CrossValidation
    ceiler = InternalConsistency()
    ceiling = ceiler(assembly)
    ceiling = ceiling.raw
    ceiling = CrossValidation().aggregate(ceiling)
    ceiling = ceiling.sel(aggregation='center')
    pass_threshold = ceiling >= threshold
    assembly = assembly[{'neuroid': pass_threshold.values}]
    return assembly


def load_logs_as_df(path, VERBOSE=False):
    from tensorboard.plugins.hparams import plugin_data_pb2
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    if VERBOSE:
        print(f'processing {path}')
    
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }

    event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
    event_acc.Reload()
    tags = event_acc.Tags()["scalars"]

    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    for tag in tags:
        event_list = event_acc.Scalars(tag)
        values = list(map(lambda x: x.value, event_list))
        step = list(map(lambda x: x.step, event_list))
        r = {"metric": [tag] * len(step), "value": values, "step": step}
        r = pd.DataFrame(r)
        runlog_data = pd.concat([runlog_data, r])

    # get hparams
    try:
        hparams_dict = plugin_data_pb2.HParamsPluginData.FromString(
            event_acc.summary_metadata['_hparams_/session_start_info'].plugin_data.content
        ).session_start_info.hparams
    except:
        print(f'Log loading failed for {path}')
        return None

    # and add to dataframe
    for hparam_key in hparams_dict:
        value = hparams_dict[hparam_key]
        # yikes, this is disgusting but it works?
        runlog_data[hparam_key] = getattr(value, str(value).split(':')[0])

    runlog_data['path'] = path

    return runlog_data
    
def multi_load_logs_as_df(paths):
    return pd.concat([load_logs_as_df(path) for path in paths])

def reduce_df(df):
    # reduce dataframes (each metric is not overlapping)
    df_ = None
    for metric in df['metric'].unique():
        df__ = df[
                (df[metric].notna())
            ].copy().reset_index().drop(columns=['index','metric']).dropna(axis='columns')
        if df_ is None:
            df_ = df__
        else:
            df_ = df_.merge(df__, how='left')

    return df_

def add_r_and_p(data, metric1, metric2, ax, x=0, y=0):
    x, y = data[metric1], data[metric2]
    
    # only take inds that are not nans from both arrays
    not_nan_inds = ~np.isnan(x)&~np.isnan(y)
    x = x[not_nan_inds]
    y = y[not_nan_inds]
    
    r, p = stats.pearsonr(x,y)
    plt.text(
        ax.get_xlim()[0], 
        ax.get_ylim()[-1], 
        f'r = {r:.2f}\np = {p:.1E}',
        verticalalignment='top',
        horizontalalignment='left',
        fontsize='x-large'
    )

def mark_baseline(data, x_metric, y_metric, ax):
    mark_x = data[
        (data['condition']=='IT + Classification')
        &(data['mix_ratio']==0)
    ][X].mean()

    mark_y = data[
        (data['condition']=='IT + Classification')
        &(data['mix_ratio']==0)
    ][plot_metric].mean()
    
    ax = sns.scatterplot(x=[mark_x], y=[mark_y], s=100, ax=ax)
    return ax

def smooth_xy_avg(data, x_metric, y_metric, ax, width=700, sample_every=5):
    xs = []
    ys = []

    for i in range(0, len(data)-width, sample_every):
        df = data.sort_values(by=[x_metric]).reset_index()[i:i+width].copy()
        xs.append(df[x_metric].mean())
        ys.append(df[y_metric].mean())
    xs = np.array(xs)
    ys = np.array(ys)

    ax = sns.lineplot(
        x=xs, y=ys, ax=ax,
        linewidth = 7,
    )
    return ax

def annotate(ax, x, y, s, style='--', c='b', ha='center', va='top'):
    ax.axhline(y=y, linestyle=style, c=c)
    ax.text(x=x, y=y, s=s, va=va, ha=ha)
    return ax

def weight_map_individual(weights, target):
    weights = eval(weights)
    if target == 'ImageNet':
        return weights[0]
    if target == 'IT':
        return weights[1]
    if target == 'HVM_labels':
        return weights[2]

def weight_map(weights):
    weights = eval(weights)
    fit = ''
    if weights[0] > 0:
        fit += 'ImageNet, '
    if weights[1] > 0:
        fit += 'IT, '
    if weights[2] > 0:
        fit += 'Stim Labels'
    return fit

condition_names = {
    'labels_0' : 'IT', 
    'labels_1' : 'IT + Classification',
    'shufflecontrol-labels_0' : 'Shuffled IT', 
    'shufflecontrol-labels_1' : 'Shuffled IT + Classification',
    'randomcontrol-labels_0' : 'Random IT',
    'randomcontrol-labels_1' : 'Random IT + Classification',
    'AT-labels_1' : 'Adv Classification', 
    'AT2-labels_1' : 'Adv Classification', 
    'ATneural-labels_1' : 'IT + Adv Classification', 
}

metric_names = {
    'ImageNet_adv_val_acc1' : 'Adversarial Accuracy (top 1)\n(PGD Linf eps = 1/1020)',
    'ImageNet_val_acc1' : 'ImageNet Validation Accuracy\n(top 1)',
    'normalized_adv_val' : 'Adversarial Accuracy / Clean Accuracy\nImageNet',
    'Stimuli_fneurons.ustimuli_val_acc1' : 'HVM var=3 Accuracy\n(top 1)',
    'Stimuli_fneurons.ustimuli_adv_val_acc1' : 'adv HVM var=3 Accuracy\n(top 1)',
    'Stimuli_magneto.var6_val_acc1' : 'HVM var=6 Accuracy\n(top 1)',
    'Stimuli_magneto.var6_adv_val_acc1' : 'adv HVM var=6 Accuracy\n(top 1)',
    'Stimuli_nano.var6_val_acc1' : 'HVM var=6 Accuracy\n(top 1)',
    'Stimuli_nano.var6_adv_val_acc1' : 'adv HVM var=6 Accuracy\n(top 1)',
    'Stimuli_bento.coco_val_acc1' : 'COCO Accuracy',
    'Stimuli_bento.coco_adv_val_acc1' : 'COCO Adversarial Accuracy',
    'CKA_fneurons.ustimuli' : 'CKA similarity\n(fitted neurons, heldout stimuli)',
    'CKA_magneto.var6' : 'CKA similarity\n(Magneto, var=6)',
    'CKA_nano.left.var6' : 'CKA similarity\n(Nano left, var=6)',
    'CKA_nano.var6_ceiled' : 'IT Neural Similarity\n(HVM, Monkey 1)',
    'CKA_magneto.var6_ceiled' : 'IT Neural Similarity\n(HVM, Monkey 2)',
    'CKA_nano.coco' : 'CKA similarity\n(Nano, COCO)',
    'CKA_bento.coco' : 'CKA similarity\n(Bento, COCO)',
    'CKA_nano.coco_ceiled' : 'IT Neural Similarity\n(COCO, Monkey 1)',
    'CKA_bento.coco_ceiled' : 'IT Neural Similarity\n(COCO, Monkey 3)',
    'CKA_adv_magneto.var6' : 'adv CKA similarity\n(Magneto, var=6)',
    'CKA_adv_nano.var6' : 'adv CKA similarity\n(Nano, var=6)',
    'dicarlo.Rajalingham2018public-i2n' : 'Rajalingham, I2N',
    'dicarlo.Rajalingham2018public-o2' : 'Rajalingham, o2',
    'dicarlo.Rajalingham2018-i2n' : 'Rajajlingham, i2n',
    'dicarlo.Rajalingham2018-o2' : 'Rajalingham, o2',
    'dicarlo.Rajalingham2018-i2n_acc' : 'Rajalingham Accuracy',
    'dicarlo.Rajalingham2018subset-i2n' : 'Rajalingham subset, i2n',
    'dicarlo.Rajalingham2018subset-i2n_acc' : 'Rajalingham subset, Accuracy',
    'dicarlo.MajajHong2015.IT-pls' : 'IT-PLS', 
    'dicarlo.MajajHong2015.V4-pls' : 'V4-PLS', 
    'movshon.FreemanZiemba2013.V2-pls' : 'V2-PLS',
    'movshon.FreemanZiemba2013.V1-pls' : 'V1-PLS',
    'HVM640.All_i1_decoder' : 'i1 HVM640 (all)',
    'HVM640.All_i1n_decoder': 'i1n HVM640 (all)', 
    'HVM640.3_i1_decoder' : 'i1 HVM640 (var=3)',
    'HVM640.3_i1n_decoder': 'i1n HVM640 (var=3)',
    'HVM640.6_i1_decoder': 'i1 HVM640 (var=6)',
    'HVM640.6_i1n_decoder': 'i1n HVM640 (var=6)',
    'dicarlo.Kar2018-i2n' : 'COCO i2n (Human)',
    'dicarlo.Kar2022human-i2n' : 'HVM640 i2n (Human)', 
    'dicarlo.Kar2022primate-i2n' : 'HVM640 i2n (Primate)',
    'dicarlo.Kar2018-i2n_acc' : 'COCO Accuracy',
    'dicarlo.Kar2022human-i2n_acc' : 'HVM640 Accuracy', 
    'dicarlo.Kar2018-o2' : 'COCO o2 (Human)',
    'dicarlo.Kar2022human-o2' : 'HVM640 o2 (Human)', 
    'dicarlo.Kar2022primate-o2' : 'HVM640 o2 (Primate)',
    'katz.BarbuMayo2019-top1' : 'ObjectNet (top 1)',
    'averaged_ceiled_CKA' : 'IT Neural Similarity\n(CKA)',
    'behavior_i2n' : 'Behavioral Alignment\n(i2n)',
    'adv_accuracy' : 'Adversarial Accuracy\n(PGD top-1)'
}
