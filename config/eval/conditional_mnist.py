import ml_collections

def get_config():

    datasets_folder = 'path/to/datasets'
    model_location = 'path/to/output/.../checkpoints/ckpt_0000999999.pt'
    model_config_location = 'path/to/output/.../config/config_001.yaml'

    config = ml_collections.ConfigDict()
    config.eval_name = 'MNIST'
    config.train_config_overrides = [
        [['device'], 'cuda'],
        [['data', 'root'], datasets_folder],
        [['distributed'], False]
    ]
    config.train_config_path = model_config_location
    config.checkpoint_path = model_location

    config.device = 'cuda'

    config.data = data = ml_collections.ConfigDict()
    data.name = 'ConditionalDiscreteMNIST'
    data.root = datasets_folder
    data.train = False
    data.download = True
    data.S = 256
    data.batch_size = 16
    data.shuffle = True
    data.shape = [1,28,28]

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = 'ConditionalImageTauLeaping' # TauLeaping or PCTauLeaping
    sampler.num_steps = 1000
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = 'gaussian'
    sampler.condition_dim = 784
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = 1.5
    sampler.corrector_entry_time = 0.1

    return config