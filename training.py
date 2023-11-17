from lampe.inference import NPE, NPELoss
from lampe.nn import MLP
from lampe.utils import GDStep
from itertools import islice
from tqdm import tqdm
from prior import Priors
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch
import zuko
import wandb

class NPEWithEmbedding(nn.Module):
    def __init__(self, 
                 num_obs, 
                 embedding_output_dim,
                 embedding_hidden_features,
                 activation,
                 transforms,
                 flow,
                 NPE_hidden_features):
        super().__init__()
        
        self.embedding = nn.Sequential(MLP(num_obs * 2,
                                           embedding_output_dim,
                                           hidden_features = embedding_hidden_features,
                                           activation = activation))
        
        self.npe = NPE(8, # The 8 parameters of an orbit
                       embedding_output_dim, 
                       transforms = transforms, 
                       build = flow, 
                       hidden_features = NPE_hidden_features, 
                       activation = activation)
        
    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        return self.npe(theta, self.embedding(x))

    def flow(self, x: Tensor): 
        return self.npe.flow(self.embedding(x))


def train(trainset, validset, epochs, num_obs, embedding_output_dim,
          embedding_hidden_features=[256] * 3, activation=nn.ELU,
          transforms=3, flow=zuko.flows.spline.NSF,
          NPE_hidden_features=[512] * 5, initial_lr=1e-3,
          weight_decay=1e-2, clip=1.0):
    """
    Train a neural posterior estimator (NPE) with an embedding network on a given dataset.
    Saves the estimator in the model file with the name of the corresponding
    wandb run.

    Args:
        trainset (Iterable): A JointLoader containing pairs of thetas and x's.
        epochs (int): The number of training epochs.
        num_obs (int): The number of observations.
        embedding_output_dim (int): The dimensionality of the output of the embedding.
        embedding_hidden_features (List[int]): A list of integers representing the number of
            hidden features in each layer of the embedding network. 
        activation (nn.Module): The activation function to use in the embedding network.
        transforms (int): The number of normalizing flow transformations to use in the NPE.
            Default is 3.
        flow (zuko.flows.Flow): The type of normalizing flow to use in the NPE. 
        NPE_hidden_features (List[int]): A list of integers representing the number of hidden
            features in each layer of the NPE. 
        initial_lr (float): The initial learning rate for the optimizer. 
        weight_decay (float): The weight decay for the optimizer.
        clip (float): The maximum gradient norm for gradient clipping. 

    Returns:
        None
    """
    prior = Priors() # Needed to preprocess the thetas

#     estimator = NPE(8,
#                     68,
#                     transforms=transforms,
#                     build = zuko.flows.spline.NSF,
#                     hidden_features=NPE_hidden_features).cuda()
    estimator = NPEWithEmbedding(
        num_obs, 
        embedding_output_dim,
        embedding_hidden_features,
        activation,
        transforms,
        flow,
        NPE_hidden_features
        ).cuda()
    
    loss = NPELoss(estimator)

    optimizer = optim.AdamW(
        estimator.parameters(), 
        lr=initial_lr, 
        weight_decay=weight_decay
        )
    
    step = GDStep(optimizer, clip=clip) 

    scheduler = sched.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        min_lr=1e-6,
        patience=32,
        threshold=1e-2,
        threshold_mode='abs'
        )
    
    wandb.init(project="SBI-Astrometry")
    config = {
        "epochs": epochs,
        "num_obs": num_obs,
        "embedding_output_dim": embedding_output_dim,
        "embedding_hidden_features": embedding_hidden_features,
        "activation": activation,
        "transforms": transforms,
        "flow": flow,
        "NPE_hidden_features": NPE_hidden_features,
        "initial_lr": initial_lr,
        "weight_decay": weight_decay,
        "clip": clip,
    }
    wandb.config.update(config)

    # theta should preprocessed in the trainset
    with tqdm(range(epochs), unit='epoch') as tq:
        for epoch in tq:

            estimator.train()
            train_loss = torch.stack([
                step(loss(prior.pre_process(theta).cuda(), x.cuda())) 
                for theta, x in islice(trainset, 1024) 
                ]).cpu().numpy()
            
            estimator.eval()
            
            with torch.no_grad():
                valid_loss = torch.stack([
                    loss(prior.pre_process(theta).cuda(), x.cuda())
                    for theta, x in islice(validset, 256) 
                    ]).cpu().numpy()

            wandb.log({
                "train_loss": train_loss.mean(), 
                "valid_loss": valid_loss.mean(), 
                "lr": optimizer.param_groups[0]['lr']
                })
            
            scheduler.step(valid_loss.mean())

            if optimizer.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
                break

            tq.set_postfix(loss=train_loss.mean(), 
                val_loss=valid_loss.mean())

    name = wandb.run.name
    torch.save(estimator, f"models/{name}.pth")
    wandb.finish()
