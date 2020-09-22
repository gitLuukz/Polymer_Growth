#
import matplotlib.pyplot as plt
import numpy as np


def chain(R,fignr):
    """
    

    Parameters
    ----------
    R : Matrix containing locations of a polymer
    fignr : Figure number.

    Returns
    -------
    Figure of polymer

    """
    plt.figure(fignr, figsize = (5,5))
    plt.plot(R[0,:],R[1,:],'k-o')
    plt.title('A polymer using the Rosenbluth algorithm ')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('../rosenbluth.png')
    return

def length(Rpolavg,N,error, fit, fignr,name):
    """
    

    Parameters
    ----------
    Rpolavg : average polymer length (weighted)
    N : Amount of beads
    error : Error of polymer length
    fit : The function made to fit the data
    fignr : Figure number
    name : Name of the figure that gets saved

    Returns
    -------
    Figure of average polymer length as function of N with error bars

    """
    plt.figure(fignr, figsize = (5,5))
    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    N = np.linspace(1,N,N)
    ax.errorbar(N,Rpolavg**2, yerr = error, fmt='b', ecolor = 'black')
    plt.plot(N,fit)
    plt.ylim(1, 10000)
    plt.xlim(2,275)
    plt.title('Length of polymer as function of beads')
    plt.xlabel('N')
    plt.ylabel('$R^2$')
    plt.grid()
    plt.savefig('../'+str(name)+'.png')
    return