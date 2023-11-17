from lampe.data import H5Dataset, JointLoader
import orbitize
from orbitize import read_input
from prior import Priors
from simulator import Simulator

def generate(loader):
    """
    Generates and stores datasets for training, validation, and testing.

    Args:
        loader: A JointLoader object.

    Returns:
        None
    """
    H5Dataset.store(
        loader, 
        f'datasets/beta-pic-test.h5', 
        size=2**17, 
        overwrite=True)
    H5Dataset.store(
        loader, 
        f'datasets/beta-pic-val.h5', 
        size=2**20, 
        overwrite=True)
    H5Dataset.store(
        loader, 
        f'datasets/beta-pic-train.h5', 
        size=2**23, 
        overwrite=True)

def main():
    data_table = read_input.read_file('{}/betaPic.csv'.format(orbitize.DATADIR))
    data_table = data_table[:-1] 

    prior = Priors()
    simulator = Simulator(data_table)
    loader = JointLoader(prior, simulator, batch_size=16, vectorized=True)
    
    generate(loader)

if __name__ == '__main__':
    main()