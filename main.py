import configs.training as config
from models.PredictorHyperNet import PredictorHyperNet
from training.utils import build_optimizer, get_device
from training.data import get_dataloader
from training.scheduler.hypernet_scheduler import build_hypernet_schedulers
from training.trainer import Trainer

# =========================
# Main
# =========================
if __name__ == "__main__":

    # Get dataloaders
    trainloader, testloader = get_dataloader(config.train_hyperparameters)
     
    # define model
    model = PredictorHyperNet()

    # Define optimizer
    optimizer = {}
    optimizer["predictor"] = build_optimizer(model.predictor, config.predictor_optimizer)
    optimizer["hypernet"] = build_optimizer(model.hypernet, config.hypernet_optimizer)

    
    # we use steps instead of epochs for scheduler to make it smoother,
    # particularly for the short warmup stage
    steps_per_epoch = len(trainloader)
    # generate schedulers as a dict with keys "predictor" and "hypernet"
    scheduler = build_hypernet_schedulers(config, optimizer, steps_per_epoch)

    # get device, cuda or cpu
    device = get_device()

    # init trainer object
    trainer = Trainer(model, trainloader, testloader, optimizer, scheduler, device = device, 
                      config = config.train_hyperparameters)
    
    # train model
    trainer.train()
