import torch
from torch.distributions import Uniform, Normal

import matplotlib.pyplot as plt
class Priors:
    """
    Class for defining the prior distributions for the parameters.
    """
    def __init__(self):
        """
        Initializes the prior distributions for the parameters.
        """
        self.LOG_LOWER = torch.log(torch.tensor(10.0))
        self.LOG_UPPER = torch.log(torch.tensor(10**4))

        self.LOWER = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        self.UPPER = torch.tensor([1.0, 180.0, 360.0, 360.0, 1.0])

        self.MEAN = torch.tensor([56.95, 1.22])
        self.STD = torch.tensor([0.26, 0.08])
        
        self.log_uniform = Uniform(self.LOG_LOWER, self.LOG_UPPER)
        self.uniform = Uniform(self.LOWER, self.UPPER)
        self.gaussian = Normal(self.MEAN, self.STD)
        
    def sample(self, ndims):
        """
        Samples from the prior distributions for the parameters.

        Args:
            ndims (int): The number of samples to generate.

        Returns:
            torch.Tensor: A tensor of shape (ndims, 8) containing the generated samples.
        """
        theta1 = torch.exp(self.log_uniform.sample(ndims))
        theta2 = self.uniform.sample(ndims)
        theta3 = self.gaussian.sample(ndims)
        
        samples = torch.cat((theta1.unsqueeze(1), theta2, theta3), dim=1)
        return samples
    
    def pre_process(self, theta):
        """
        Pre-processes the generated samples.

        Args:
            theta (torch.Tensor): A tensor of shape (ndims, 8) containing the generated samples.

        Returns:
            torch.Tensor: A tensor of shape (ndims, 8) containing the pre-processed samples.
        """
        theta1, theta2, theta3 = theta[:, 0], theta[:, 1:6], theta[:, 6:8]

        theta1 = 2 * (torch.log(theta1) - self.LOG_LOWER) / (self.LOG_UPPER - self.LOG_LOWER) - 1
        theta2 = 2 * (theta2 - self.LOWER) / (self.UPPER - self.LOWER) - 1
        theta3 = (theta3 - self.MEAN) / (self.STD*3) # 3 times the standard deviation to get 99.7% of the data between -1 and 1

        theta = torch.cat((theta1.unsqueeze(1), theta2, theta3), dim=1)

        return theta
    
    def post_process(self, theta):
        """
        Post-processes the generated samples.

        Args:
            theta (torch.Tensor): A tensor of shape (ndims, 8) containing the generated samples.

        Returns:
            torch.Tensor: A tensor of shape (ndims, 8) containing the post-processed samples.
        """
        theta1, theta2, theta3 = theta[:, 0], theta[:, 1:6], theta[:, 6:8]

        theta1 = torch.exp((theta1 + 1) * (self.LOG_UPPER - self.LOG_LOWER) / 2 + self.LOG_LOWER)
        theta2 = (theta2 + 1) * (self.UPPER - self.LOWER) / 2 + self.LOWER
        theta3 = theta3 * self.STD*3 + self.MEAN

        theta = torch.cat((theta1.unsqueeze(1), theta2, theta3), dim=1)

        return theta
