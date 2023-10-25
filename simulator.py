import torch
import numpy as np

from orbitize.system import seppa2radec
from orbitize.kepler import calc_orbit

class OrbitCreator:
    def __init__(self, data_table):
        self.data_table = data_table
        self.number_epochs = len(data_table)

    def __call__(self, thetas: torch.Tensor) -> torch.Tensor:
        n_samples, _ = thetas.shape

        results = torch.empty((n_samples,
                               2 * self.number_epochs),
                               dtype=torch.float32)

        for i in range(n_samples):
            theta = thetas[i]
            sma, ecc, inc, aop, pan, tau, plx, mtot = theta.numpy()

            ra, dec, _ = calc_orbit(
                self.data_table["epoch"], sma, ecc, inc, aop, pan,
                tau, plx, mtot, tau_ref_epoch=51544, use_c=False
            )

            ra = torch.tensor(ra,dtype=torch.float32)
            dec = torch.tensor(dec,dtype=torch.float32)

            # Adding Gaussian offets from observational uncertainties
            ra_err, dec_err = seppa2radec(self.data_table["quant1_err"], 
                                          self.data_table["quant2_err"])
            ra_err = np.random.normal(0, ra_err)
            dec_err = np.random.normal(0, dec_err)

            result_tensor = torch.empty(2 * len(ra), dtype=torch.float32)
            result_tensor[0::2] = ra + ra_err
            result_tensor[1::2] = dec + dec_err
            results[i] = result_tensor

        return results
    
    