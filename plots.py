import matplotlib.pyplot as plt
from astropy.time import Time
from orbitize.system import seppa2radec
from prior import Priors

def ra_dec_plot(ra, dec, data_table):
    
    observation_epochs = data_table['epoch'].data
    
    fig, ax = plt.subplots() 
    
    ra_obs, dec_obs = seppa2radec(data_table['quant1'].data,
                         data_table['quant2'].data)
     
        
        
    shw1 = ax.scatter(ra_obs, dec_obs, c=Time(observation_epochs, format='mjd').decimalyear, cmap='Blues', label="Observed data")
    shw2 = ax.scatter(ra, dec, c=Time(observation_epochs, format='mjd').decimalyear, cmap='Greens', label="Simulated data")

    bar1 = plt.colorbar(shw1)
    bar2 = plt.colorbar(shw2)
   
    bar2.set_ticks([])

    plt.xlabel('$\\Delta$ RA'); plt.ylabel('$\\Delta$ Dec')
    plt.axis('equal')
    plt.plot(0,0,marker="*",color='black',markersize=10)
    bar1.set_label('Year')
    plt.legend()
    plt.show()
    
def corner(estimator, data_table):
    prior = Priors()
    
    quant1 = data_table["quant1"].data
    quant2 = data_table["quant2"].data

    ra, dec = seppa2radec(quant1, quant2)

    x_star = torch.empty(2 * len(ra), dtype=torch.float32)
    x_star[0::2] = torch.tensor(ra, dtype=torch.float32)
    x_star[1::2] = torch.tensor(dec, dtype=torch.float32)

    x_star = x_star.clone().detach()
    
    with torch.no_grad():
        samples = estimator.flow(x_star.cuda()).sample((2**16,)).cpu()
        samples = prior.post_process(samples)
        
    LOWER = torch.tensor([10.0, 0, -10.0, -10.0, -10.0, 0.0, 55.91, 0.9])
    UPPER = torch.tensor([10000, 1, 180.0, 360.0, 360.0, 1.0, 57.99, 1.54])

    LABELS = [r'$a$', r'$e$', r'$i$',
            r'$\omega$', r'$\Omega$',
            r'$\tau$', r'$\pi$', r'$M_T$']

    plt.rcParams.update(nice_rc(latex=True))  # nicer plot settings

    fig = corner(
        samples,
        smooth=2,
        labels=LABELS,
        legend=r'$p_\phi(\theta | x^*)$',
    )
    