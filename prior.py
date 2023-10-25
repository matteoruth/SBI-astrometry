import torch
from torch.distributions import Uniform, Normal

class prior_distributions():
    def __init__(self, 
                 log_uniform_lower, 
                 log_uniform_upper, 
                 uniform_lower,
                 uniform_upper, 
                 gaussian_mean, 
                 gaussian_std):

        self.log_uniform_lower = torch.log(log_uniform_lower)
        self.log_uniform_upper = torch.log(log_uniform_upper)

        self.uniform_lower = uniform_lower
        self.uniform_upper = uniform_upper

        self.gaussian_mean = gaussian_mean
        self.gaussian_std = gaussian_std

        self.log_uniform = Uniform(self.log_uniform_lower, self.log_uniform_upper)
        self.uniform = Uniform(self.uniform_lower, self.uniform_upper)
        self.gaussian = Normal(gaussian_mean, gaussian_std)

    def sample(self,  ndims):
        theta1 = torch.exp(self.log_uniform.sample(ndims))
        theta2 = self.uniform.sample(ndims)
        theta3 = self.gaussian.sample(ndims)

        samples = torch.cat((theta1.unsqueeze(1), theta2, theta3), dim=1)
        return samples
    
    def log_prob(self, samples):
        theta1, theta2, theta3 = samples[:, 0], samples[:, 1:6], samples[:, 6:8]
        log_prob_theta1 = self.log_uniform.log_prob(torch.log(theta1)).sum()
        log_prob_theta2 = self.uniform.log_prob(theta2).sum()
        log_prob_theta3 = self.gaussian.log_prob(theta3).sum()

        total_log_prob = log_prob_theta1 + log_prob_theta2 + log_prob_theta3
        return total_log_prob