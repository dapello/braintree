##  some imports wrapped in function calls to avoid making brainscore and model_tools strictly necessary.
import h5py as h5
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from scipy.stats import zscore, norm
from sklearn import preprocessing

NEURAL_DATA_PATH = '/om2/user/dapello'

def wrap_model(identifier, model, image_size):
    import functools
    from model_tools.activations.pytorch import PytorchWrapper
    from model_tools.activations.pytorch import load_preprocess_images
    
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size, normalize_mean=(0,0,0), normalize_std=(1,1,1))
    wrapper = PytorchWrapper(identifier=identifier, model=model, preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper

def cornet_s_brainmodel(identifier, model, image_size):
    import functools
    from candidate_models.base_models.cornet import TemporalPytorchWrapper
    from candidate_models.model_commitments.cornets import CORnetCommitment, CORNET_S_TIMEMAPPING, _build_time_mappings
    from model_tools.activations.pytorch import load_preprocess_images

    # map region -> (time_start, time_step_size, timesteps)
    time_mappings = CORNET_S_TIMEMAPPING

    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    wrapper = TemporalPytorchWrapper(identifier=identifier, model=model[1].module, preprocessing=preprocessing, separate_time=True)
    wrapper.image_size = image_size

    return CORnetCommitment(identifier=identifier, activations_model=wrapper,
                            layers=['V1.output-t0'] +
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                                    for timestep in timesteps] +
                                   ['decoder.avgpool-t0'],
                            time_mapping=_build_time_mappings(time_mappings))

def brain_wrap_model(identifier, model, layers, image_size):
    from model_tools.brain_transformation import ModelCommitment

    activations_model = wrap_model(identifier, model, image_size)
    
    brain_model = ModelCommitment(identifier=identifier, activations_model=activations_model, 
        layers=layers)
    
    return brain_model

def score_model(model_identifier, model, benchmark_identifier, layers=[], image_size=224):
    import os 
    from brainscore import score_model as _score_model
    os.environ['RESULTCACHING_DISABLE'] = 'brainscore.score_model,model_tools'

    if 'cornet' in model_identifier:
        brain_model = cornet_s_brainmodel(identifier=model_identifier, model=model, image_size=image_size)
    else:
        brain_model = brain_wrap_model(identifier=model_identifier, model=model, layers=layers, image_size=224)

    score = _score_model(
        model_identifier=model_identifier, model=brain_model, 
        benchmark_identifier=benchmark_identifier
    )

    return score

def list_brainscore_benchmarks():
    benchmarks = []
    try:
        import brainscore
        benchmarks += brainscore.benchmark_pool.keys()
    except:
        print(">>> Brainscore benchmarks not accessible.")

    return benchmarks

### Ko i1 benchmark

class BehaviorScorer:

    # path to stimuli
    data = h5.File(f'{NEURAL_DATA_PATH}/neural_data/many_monkeys2.h5', 'r')

    def __init__(self, variations=6, seeds=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # how many seeds to run when computing i1
        self.seeds = seeds
        
        # which HVM variations to consider
        if variations == 'All':
            # return all stimuli
            self.idxs = np.array(range(len(self.data['var'][()])))
        if variations == 3:
            self.idxs = self.data['var'][()] == 3
        if variations == 6:
            self.idxs = self.data['var'][()] == 6

    def get_stimuli(self):
        # loads stimuli (as np array) filtered by HVM variations
        X = self.data['stimuli'][()][self.idxs].transpose(0,3,1,2)
        return X
    
    def get_labels(self):
        # constructs labels, filters by HVM variations
        lb = ['bear', 'elephant', 'faces', 'car', 'dog', 'apple', 'chair', 'plane']
        labels = np.repeat(lb, 80, axis=0)
        labels = labels[self.idxs]
        return labels, lb

    def get_target(self, metric):
        # fetches the appropriate primate i1 / i1n score
        if metric == 'i1':
            target = h5.File(
                f'{NEURAL_DATA_PATH}/neural_data/i1_hvm640.mat', 'r+'
            )['i1_hvm640']
        elif metric == 'i1n':    
            target = h5.File(
                f'{NEURAL_DATA_PATH}/neural_data/i1n_hvm640.mat', 'r+'
            )['i1n_hvm640']
            
        # filters by HVM variations
        target = target[()].squeeze()[self.idxs]
        
        return target
                
    def get_features(self, model_identifier, model, layer, image_size):
        wrapped_model = wrap_model(
            identifier=model_identifier,
            model=model, 
            image_size=image_size
        )
        
        stimuli = self.get_stimuli()
        
        features = wrapped_model.get_activations(
            stimuli, layer_names=[layer]
        )[layer].squeeze()

        features = features.reshape(features.shape[0], -1)
        
        return features
    
    def score_model(self, metric, model_id, model, layer, image_size=224):
        features = self.get_features(model_id, model, layer, image_size)
        
        if metric == 'i1':
            prediction = self.compute_i1(features)
            
        if metric == 'i1n':
            prediction = self.compute_i1(features)
            prediction = self.compute_i1n(prediction)
        
        target = self.get_target(metric)
        
        pearsons_r = np.corrcoef(target, prediction)[0,1]
        
        return pearsons_r
        
    def compute_i1(self, features):
        features = features.T
        nrImages = features.shape[1] # number of images retrieved from the feature matrix
        i_1 = np.zeros((nrImages,1,self.seeds), dtype=float)
        i_1[:]=np.NAN
        
        labels, lb = self.get_labels()

        for j in range(self.seeds):
            p = decode(features, labels, seed=j, nrfolds=3)
            pc = get_percent_correct_from_proba(p, labels, np.array(lb))
            fa, full_fa = get_fa(pc, labels)
            dp = get_dprime(np.nanmean(pc,axis=1),fa)
            i_1[:,0,j] = dp
            
        i1_mean = np.mean(i_1,axis=2).squeeze()
        
        return i1_mean
    
    def compute_i1n(self, i1):
        object_means = i1.reshape(8,-1).mean(axis=1)
        i1n = (i1.reshape(8,-1).T - object_means).T.reshape(-1)
        
        return i1n

def score_model_behavior(model_id, model, layer, benchmark):
    dataset = benchmark.split('_')[0]
    metric = benchmark.split('_')[1]
    if '.All' in dataset:
        behavior_scorer = BehaviorScorer(variations='All')
    elif '.3' in dataset:
        behavior_scorer = BehaviorScorer(variations=3)
    elif '.6' in dataset:
        behavior_scorer = BehaviorScorer(variations=6)
    
    score = behavior_scorer.score_model(
        metric=metric,
        model_id=model_id,
        model=model,
        layer=layer,
    )
    
    return score

def decode(features,labels,nrfolds=2,seed=0):
    classes=np.unique(labels)
    nrImages = features.shape[1]
    _,ind = np.unique(classes, return_inverse=True)   

    #scale data
    features = zscore(features, axis=0)
    num_classes = len(classes)
    prob = np.zeros((nrImages,len(classes)))
    prob[:]=np.NAN

    for i in range(nrfolds):
        train, test = get_train_test_indices(
            nrImages,nrfolds=nrfolds, foldnumber=i, seed=seed
        )
        
        XTrain = features[:,train]
        XTest = features[:,test]
        YTrain = labels[train]
        
        clf = LogisticRegression(
            penalty='l2', C=5*10e4, multi_class='ovr', 
            max_iter=1000, class_weight='balanced',
            solver='liblinear'
        )
        clf.fit(XTrain.T, YTrain)
        
        pred=clf.predict_proba(XTest.T)
        prob[test,0:num_classes] = pred

    return prob

def get_percent_correct_from_proba(prob, labels, class_order):
    nrImages = prob.shape[0]
    class_order=np.unique(labels)
    pc = np.zeros((nrImages,len(class_order)))
    pc[:]=np.NAN
    _,ind = np.unique(labels, return_inverse=True)
    
    for i in range(nrImages):
        loc_target = labels[i]==class_order
        pc[i,:] = np.divide(
            prob[i,labels[i]==class_order], 
            prob[i,:]+prob[i,loc_target]
        )
        pc[i,loc_target] = np.NAN

    return pc

def get_fa(pc, labels):
    _, ind = np.unique(labels, return_inverse=True)
    full_fa = 1-pc
    pfa = np.nanmean(full_fa,axis=0)
    fa = pfa[ind]    

    return fa, full_fa

def get_dprime(pc,fa):
    zHit = norm.ppf(pc)
    zFA = norm.ppf(fa)
    
    # controll for infinite values
    zHit[np.isposinf(zHit)] = 5
    zFA[np.isneginf(zFA)] = -5

    # Calculate d'
    dp = zHit - zFA
    dp[dp>5]=5
    dp[dp<-5]=-5

    return dp

def get_train_test_indices(totalIndices, nrfolds=10, foldnumber=0, seed=1):
    """
    Parameters
    ----------
    totalIndices : TYPE
        DESCRIPTION.
    nrfolds : TYPE, optional
        DESCRIPTION. The default is 10.
    foldnumber : TYPE, optional
        DESCRIPTION. The default is 0.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    train_indices : TYPE
        DESCRIPTION.
    test_indices : TYPE
        DESCRIPTION.
    """
    
    np.random.seed(seed)
    inds = np.arange(totalIndices)
    np.random.shuffle(inds)
    splits = np.array_split(inds,nrfolds)
    test_indices = inds[np.isin(inds,splits[foldnumber])]
    train_indices = inds[np.logical_not(np.isin(inds, test_indices))]
    return train_indices, test_indices

def list_behavior_benchmarks():
    benchmarks = [
        'HVM640.All_i1',
        'HVM640.All_i1n',
        'HVM640.3_i1',
        'HVM640.3_i1n',
        'HVM640.6_i1',
        'HVM640.6_i1n'
    ]

    return benchmarks
