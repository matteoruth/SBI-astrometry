import matplotlib.pyplot as plt
from astropy.time import Time

def ra_dec_plot(ra, dec, observation_epochs):
    plt.figure()
    plt.scatter(ra, dec, c=Time(observation_epochs, format='mjd').decimalyear, cmap='winter')

    plt.xlabel('$\\Delta$ RA'); plt.ylabel('$\\Delta$ Dec')
    plt.axis('equal')
    plt.plot(0,0,marker="*",color='black',markersize=10)
    plt.colorbar(label='Year')
    plt.show()