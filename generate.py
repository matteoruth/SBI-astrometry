from lampe.data import JointLoader, H5Dataset
from time import time
import torch
import orbitize
from orbitize import read_input
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from prior import prior_distributions
from simulator import OrbitCreator

def simulate(loader, term, size):
        H5Dataset.store(
            loader, 
            f'Data/{term}/beta-pic-{term}-8.h5', 
            size=(size), 
            overwrite=True)
    
def generate(loader):
    #simulate(loader, 'test', 2**17)
    #simulate(loader, 'val', 2**20) 
    simulate(loader, 'train', 2**20) # faire ça 8 fois pour un total de 2**23
    
class data_processing:
  def __init__(self, trainset, scaler1=MinMaxScaler, scaler2 = MinMaxScaler):

    thetas = trainset[:][0]
    xs = trainset[:][1]

    self.scaler_theta = scaler1()
    self.scaler_x = scaler2()

    self.thetas_processed = self.scaler_theta.fit(thetas)
    self.xs_processed = self.scaler_x.fit(xs)

  def preprocess_x(self, x):
    return torch.Tensor(self.scaler_x.transform(x))

  def preprocess_theta(self, theta):
    return torch.Tensor(self.scaler_theta.transform(theta))

  def postprocess_x(self, x):
    return torch.Tensor(self.scaler_x.inverse_transform(x))

  def postprocess_theta(self, theta):
    return torch.Tensor(self.scaler_theta.inverse_transform(theta))


def main():
    data_set = read_input.read_file('{}/betaPic.csv'.format(orbitize.DATADIR))
    prior = prior_distributions(log_uniform_lower = torch.tensor(10.0), 
                                log_uniform_upper = torch.tensor(10**4),
                                uniform_lower = torch.tensor([10e-8, 0.0, 0.0, 0.0, 0.0]), 
                                uniform_upper = torch.tensor([0.99, 180.0, 360.0, 360.0, 1.0]),
                                gaussian_mean = torch.tensor([56.95, 1.22]), 
                                gaussian_std = torch.tensor([0.26, 0.08]))
    simulator = OrbitCreator(data_set)
    loader = JointLoader(prior, simulator, batch_size=16, vectorized=True)

    generate(loader)

if __name__ == "__main__":
    main()
    
