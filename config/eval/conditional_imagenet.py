import ml_collections

def get_config():

    datasets_folder = 'path/to/imagenet/2012'
    model_location = 'path/to/imagenet_output/.../ckpt_0000200000.pt'
    model_config_location = 'path/to/.../config/config_001.yaml'

    config = ml_collections.ConfigDict()
    config.eval_name = 'ImageNet'
    config.train_config_overrides = [
        [['device'], 'cuda'],
        [['data', 'root'], datasets_folder],
        [['distributed'], False]
    ]
    config.train_config_path = model_config_location
    config.checkpoint_path = model_location

    config.device = 'cuda'

    config.data = data = ml_collections.ConfigDict()
    data.name = 'ConditionalDiscreteImageNet'
    data.root = datasets_folder
    data.train = False
    data.S = 256
    data.batch_size = 1
    data.shuffle = False
    data.shape = [3,256,256]
    data.random_flips = True
    data.high_resolution = 256
    data.low_resolution = 64

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = 'ConditionalImageTauLeapingWithLabel' # TauLeaping or PCTauLeaping
    sampler.num_steps = 100
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = 'gaussian'
    sampler.condition_dim = 3*256*256
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = 1.5
    sampler.corrector_entry_time = 0.1

    return config