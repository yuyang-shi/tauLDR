import ml_collections

def get_config():
    save_directory = 'path/to/imagenet_output' 
    datasets_folder = '/jmain02/home/J2AD008/wga35/yxs83-wga35/datasets/imagenet/2012'

    config = ml_collections.ConfigDict()
    config.experiment_name = 'conditional_imagenet'
    config.save_location = save_directory

    config.init_model_path = None
    config.pretrained_ckpt = '/jmain02/home/J2AD008/wga35/yxs83-wga35/tauLDR/64_256_upsampler.pt'

    config.device = 'cuda'
    config.distributed = False
    config.num_gpus = 1

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = 'ConditionalAuxWithLabel'
    loss.eps_ratio = 1e-9
    loss.nll_weight = 0.001
    loss.min_time = 0.01
    loss.condition_dim = 3*256*256
    loss.one_forward_pass = True

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = 'Standard'
    training.n_iters = 2000000
    training.clip_grad = True
    training.warmup = 5000

    config.data = data = ml_collections.ConfigDict()
    data.name = 'ConditionalDiscreteImageNet'
    data.root = datasets_folder
    data.train = True
    data.S = 256
    data.batch_size = 2 # use 128 if you have enough memory or use distributed
    data.shuffle = True
    data.shape = [3,256,256]
    data.random_flips = True
    data.high_resolution = 256
    data.low_resolution = 64
    data.num_workers = 4

    config.model = model = ml_collections.ConfigDict()
    model.name = 'ConditionalGaussianTargetRateImageX0PredWithLabelEMA'

    model.ema_decay = 0.9999 #0.9999

    model.ch = 192
    model.num_res_blocks = 2
    model.ch_mult = [1,1,2,2,4,4]
    model.input_channels = 3
    model.attention_resolutions = [32,16,8]
    model.data_min_max = [0, 255]
    model.dropout = 0
    model.use_checkpoint = True
    model.fix_logistic = False

    model.rate_sigma = 6.0
    model.Q_sigma = 512.0
    model.time_exponential = 100.0
    model.time_base = 3.0


    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = 'Adam'
    optimizer.lr = 1e-4
    optimizer.weight_decay = 0

    config.saving = saving = ml_collections.ConfigDict()

    saving.enable_preemption_recovery = False
    saving.preemption_start_day_YYYYhyphenMMhyphenDD = None
    saving.checkpoint_freq = 1000
    saving.num_checkpoints_to_keep = 2
    saving.checkpoint_archive_freq = 200000
    saving.log_low_freq = 10000
    saving.low_freq_loggers = ['ConditionalDenoisingImagesWithLabel']
    saving.prepare_to_resume_after_timeout = False

    return config
