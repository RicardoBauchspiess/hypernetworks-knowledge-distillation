
train_hyperparameters = {
    "batch_size":128,
    "augmentation":"basic",
    "epochs":40,
    "head_start":3 # epochs only training hypernet
}

predictor_optimizer = {
    "optimizer":"sgd",
    "lr":0.05,
    "weight_decay":5e-4,
    "nesterov":False,
    }

hypernet_optimizer = {
    "optimizer":"sgd",
    "lr":0.05,
    "weight_decay":5e-4,
    "nesterov":False,
    }

predictor_scheduler = {
    "scheduler":"warmup_cosine",
    "warmup_steps":800,
    "total_steps":0, # calculated externaly based on number of epochs and dataloader lenght
    }

hypernet_scheduler = {
    "scheduler":"warmup_cosine",
    "warmup_steps":800,
    "total_steps":0, # calculated externaly based on number of epochs and dataloader lenght
    }