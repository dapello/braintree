# imports wrapped in function calls to avoid making brainscore and model_tools strictly necessary.

def wrap_model(identifier, model, image_size):
    import functools
    from model_tools.activations.pytorch import PytorchWrapper
    from model_tools.activations.pytorch import load_preprocess_images
    
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size, normalize_mean=(0,0,0), normalize_std=(1,1,1))
    wrapper = PytorchWrapper(identifier=identifier, model=model, preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper

def brain_wrap_model(identifier, model, layers, image_size):
    from model_tools.brain_transformation import ModelCommitment

    activations_model = wrap_model(identifier, model, image_size)
    
    brain_model = ModelCommitment(identifier=identifier, activations_model=activations_model, 
        layers=layers)
    
    return brain_model

def score_model(model_identifier, model, layers, benchmark_identifier, image_size=224):
    import os 
    from brainscore import score_model as _score_model
    os.environ['RESULTCACHING_DISABLE'] = 'brainscore.score_model,model_tools'

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

# dict_keys(['movshon.FreemanZiemba2013public.V1-pls', 'movshon.FreemanZiemba2013public.V2-pls', 'dicarlo.MajajHong2015public.V4-pls', 'dicarlo.MajajHong2015public.IT-pls', 'dicarlo.Rajalingham2018public-i2n', 'fei-fei.Deng2009-top1', 'dietterich.Hendrycks2019-noise-top1', 'dietterich.Hendrycks2019-blur-top1', 'dietterich.Hendrycks2019-weather-top1', 'dietterich.Hendrycks2019-digital-top1', 'dicarlo.MajajHong2015.V4-mask', 'dicarlo.MajajHong2015.IT-mask', 'dicarlo.MajajHong2015.V4-rdm', 'dicarlo.MajajHong2015.IT-rdm', 'movshon.FreemanZiemba2013.V1-rdm', 'movshon.FreemanZiemba2013.V2-rdm', 'movshon.FreemanZiemba2013.V1-single', 'tolias.Cadena2017-pls', 'tolias.Cadena2017-mask', 'dicarlo.Sanghavi2020.V4-pls', 'dicarlo.Sanghavi2020.IT-pls', 'dicarlo.SanghaviJozwik2020.V4-pls', 'dicarlo.SanghaviJozwik2020.IT-pls', 'dicarlo.SanghaviMurty2020.V4-pls', 'dicarlo.SanghaviMurty2020.IT-pls', 'dicarlo.Rajalingham2020.IT-pls', 'dicarlo.MajajHong2015.V4-pls', 'dicarlo.MajajHong2015.IT-pls', 'movshon.FreemanZiemba2013.V1-pls', 'movshon.FreemanZiemba2013.V2-pls', 'dicarlo.Kar2019-ost', 'dicarlo.Rajalingham2018-i2n']))
