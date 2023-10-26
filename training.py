import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from itertools import islice
from lampe.utils import GDStep
from tqdm import tqdm

def train_sbi(name,
              estimator, 
              estimator_loss,
              trainset, 
              validset, 
              epochs = 128):
    
    loss = estimator_loss(estimator)
    optimizer = optim.AdamW(estimator.parameters(), 
                            lr=1e-3, 
                            weight_decay=1e-2)
    step = GDStep(optimizer, clip=1.0) 
    scheduler = sched.ReduceLROnPlateau(optimizer,
                                        factor=0.5,
                                        min_lr=1e-6,
                                        patience=32,
                                        threshold=1e-2,
                                        threshold_mode='abs')
    
    # # Check if wandb can be used on the cluster
    # wandb.init(project="SBI-Orbitize-NPE",
    #            config={"Optimizer" : ,
    #                    "Learning_rate": ,
    #                    "Weight_decay" : ,
    #                    "Gradient descent cliping" : ,
    #                    "Transforms": ,
    #                    "Structure" : ,
    #                    "Training set size" : ,
    #                    "Epochs": ,
    #                    "Standardized" : ,
    #                    "Scheduler" : ,
    #                    "Factor" : ,
    #                    "Min_lr" : ,
    #                    "Patience" : ,
    #                    "Threshold" : ,
    #                    "Threshold_mode" : })

    with tqdm(range(epochs), unit='epoch') as tq:
        for epoch in tq:
            estimator.train()

            train_loss = torch.stack([
                
                step(loss(theta.cuda(), x[:, :-2].cuda())) # x[:, :-2] to remove rv data, should recreate a dataset
                for theta, x in islice(trainset, 1024)
            ]).cpu().numpy()

            estimator.eval()
            with torch.no_grad():
                valid_loss = torch.stack([
                    loss(theta.cuda(), x[:, :-2].cuda()) # x[:, :-2] to remove rv data, should recreate a dataset
                    for theta, x in islice(validset, 256)
                ]).cpu().numpy()

            scheduler.step(valid_loss.mean())

            # # Check if wandb can be used on the cluster
            # wandb.log({'lr': optimizer.param_groups[0]['lr'],
            #         'training loss': np.nanmean(train_epoch_losses),
            #         'validation loss': np.nanmean(val_epoch_losses),
            #         'nans': np.isnan(train_epoch_losses).mean(),
            #         'nans_val': np.isnan(val_epoch_losses).mean()})
            tq.set_postfix(loss=train_loss.mean(), 
                           val_loss=valid_loss.mean())
    # # Check if wandb can be used on the cluster
    # wandb.finish()
    torch.save(estimator.state_dict(), f'Models/{name}.pth')
    