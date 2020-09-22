"""
Functions of this project
"""

import numpy as np
from scipy.optimize import curve_fit

def distance(R,L):
    """
    Parameters
    ----------
    R : 
        Matrix of positions in 2D
    L : TYPE
        Current particle nr

    Returns
    -------
    dis :
        Distance matrix
    sq_dis :
        squared distance that can be used for lennard jones potential
    """
    dis  = (R[:,:L].T - R[:,L]).T
    r = np.sqrt(np.sum(dis**2,0))
    return dis, r


def LJ(L,r,epsilon,sig):
    """
    Parameters
    ----------
    L : 
        Last particle number
    r :
        Distance between particle L and the rest
    epsilon : 
        Constant
    sig : 
        Constant

    Returns
    -------
    U : TYPE
        Potential between the particle L+1 and particles 1-(L) calculated after distance
    """
    U = np.sum(4*epsilon*((sig/r)**12-(sig/r)**6))
    return U

def weights(n,L,R,epsilon,sig):
    """
    Parameters
    ----------
    n :
        Number of new angles tried    
    L :
        Last particle number  
    R :
       Distance between the beads  
    epsilon : 
        Constant
    sig : 
        Constant

    Returns
    -------
    W :
        The sum of the weights of the six possible angles
    w :
        The weight corresponding to a certain angle theta
    theta :	TYPE
        The angle at which the new bead is placed
    """

    w = np.zeros(n)  
    theta = np.arange(n)*2*np.pi/n + np.random.uniform(0,1) 
    for i in range(len(w)):
        R[:,L+1] = R[:,L]+np.array([np.cos(theta[i]),np.sin(theta[i])])
        [dis,r] = distance(R, L+1)
        U = LJ(L+1, r, epsilon, sig)
        w[i] = np.exp(-U)
    W = np.sum(w)
    
    return W,w,theta


def roulete(w):
    """
    Parameters
    ----------
    w :
        The weight corresponding to a certain angle theta

    Returns
    -------
    index :
        The index corresponding to the weight 
    """

    test_value = np.random.uniform(0,w[-1])
    
    index = np.digitize(test_value,w[0:-1])
    return index

def simulation(R,L,n,epsilon,sig,N,Wsave,wsave):
    """
    Parameters
    ----------
    R :
        Distance between the beads  
    L :
        Last particle number  
    n :
        Number of new angles tried
    epsilon : 
        Constant
    sig : 
        Constant
    N : 
        Constant
    Wsave :
        Initialisation
    wsave :
        Initialisation
        
    Returns
    -------
    R :
        Distance between the beads
    Wsave :
        The sum of the weights of the six possible angles
    wsave :
        The weight corresponding to a certain angle theta

    """

    W,w,theta = weights(n, L, R, epsilon, sig)
    w2 = np.cumsum(w)
    index = roulete(w2)
        
    wsave[L-1] = w[int(index)]
    Wsave[L-1] = W
    
    theta_choice = theta[int(index)]
    R[:,L+1] = R[:,L]+np.array([np.cos(theta_choice),np.sin(theta_choice)])
    L = L+1
    
    if L<N-1:
        simulation(R, L, n, epsilon, sig,N,Wsave,wsave)
    return R , wsave, Wsave



def Iterative_simulation(Group_size,N,epsilon,sig,R_save,distance_save,w_save,W_save):
    """
    Parameters
    ----------
    Groupsize : 
        Constant
    N : 
        Constant
    epsilon : 
        Constant
    sig : 
        Constant 
    R_save : 
        Initialisation  
    Distance_save :
        Initialisation  
    w_save : 
        Initialisation
    W_save :
        Initialisation
        
    Returns
    -------
    R_save :
        Distance between the beads
    Distance_save :
        The end-to-end distance
    w_save :
        The weight corresponding to a certain angle theta
    W_save :
        The sum of the weights of the six possible angles
    """

    for i in range(Group_size):
        L = 1  # indices of the particle that is last made.
        n = 6  # number of new angeles tried
        
        
        R = np.zeros((2,N))
        R[0,1] = 1
        
        wsave = np.zeros(N-2)
        Wsave = np.zeros(N-2)
        
        
        R , wsave, Wsave = simulation(R, L, n, epsilon, sig, N, Wsave, wsave)
        
        distance = np.sqrt(np.sum(R**2,0))
        
        R_save[:,:,i] = R
        distance_save[:,:,i] = distance
        w_save[:,:,i] = wsave
        W_save[:,:,i] = Wsave
        
        
    return R_save, distance_save,w_save,W_save

def modify(P_save,Rpol_avg,N,Group_size,distance_save):
    """
    Parameters
    ----------
    P_save : 
        The probability of the polymer occuring        
    Rpol_avg :
        Initialisation
    N : 
        Constant
    Group_size : 
        Constant
    distance_save :
         The end-to-end distance
         
    Returns
    -------
    Rpol_avg :
        The weighted average end-to-end distance. 
    """
    
    Ptrans = np.zeros((Group_size,N-2))
    Ptrans = P_save.transpose()
    Rpol_avg[0,0] = 0;
    Rpol_avg[1,0] = 1;
    
    for j in range(N-2):
        Rpol_avg[j+2,0] = np.dot(distance_save[0,j+2,:],Ptrans[:,j])/np.sum(Ptrans[:,j])
    return Rpol_avg

def sigma(N,P_save,Group_size,distance_save):
    """
    Parameters
    ----------
    N : 
        Constant
    P_save : 
        The probability of the polymer occuring
    Group_size : 
        Constant
    distance_save : 
         The squared end-to-end distance
         
    Returns
    -------
    sigma_A :	
        The weighted error of the end-to-end distance 
    """
    
    sigma_A = np.zeros(N)
    sigma_A[0] = 0
    sigma_A[1] = 0
    
    for j in range(N-2):
        sigma_A[j+2] = 1/np.sqrt(np.count_nonzero(P_save[j,:]))*np.sqrt(1/(np.count_nonzero(P_save[j,:])-1)*np.sum((distance_save[0,j+2,:]- np.dot(distance_save[0,j+2,:],P_save[j,:])/np.sum(P_save[j,:]))**2))
    return sigma_A

def fit(N,Rpol_avg):
    """
    Parameters
    ----------
    N : 
        Constant
   Rpol_avg : 
        The weighted average end-to-end distance

    Returns
    -------
    popt :	
        Fitting parameter        
    """
    Nvec = np.linspace(1,N,N)
    popt = curve_fit(fit_function,Nvec,Rpol_avg)
    return popt
    
    
def fit_function(N,a):
    """
    Parameters
    ----------
    N : 
        Constant
   a : TYPE
        Initialising

    Returns
    -------
       Fitted curve with fitted parameter a        
    """
    return (a*(N-1)**0.75)**2




def PERM(size,epsilon,sigma,N):
    """

    Parameters
    ----------
    size :
        constant for the size of matrices used
    epsilon :
        constant
    sigma :
        constant
    N :
        length of polymers
    Returns
    -------
    R_new :
        Return polymers per iteration
    wsave_new :
        Return weights per iteration
    Wsave_new :
        Weights per iteration (Sum of possible weights)

    """
    poly_num = 0
    L = 1 
    n = 6  
    
    R_new = np.zeros((2,N,size))
    R_new[0,1,0] = 1
    wsave_new = np.zeros((N-2,size))
    Wsave_new = np.zeros((N-2,size))
    
    const = 1
    poly_num, R_new , wsave_new ,Wsave_new= simulation_perm(R_new, L, n, epsilon, sigma, N, wsave_new,Wsave_new, poly_num,const)
    
    return R_new, wsave_new,Wsave_new

def simulation_perm(R,L,n,epsilon,sigma,N,wsave,Wsave,poly_num,const):
    """
    Parameters
    ----------
    R : TYPE
        Matrix that will be filled with polymer locations
    L : TYPE
        Indice that indicates polymer lenght in simulation
    n : TYPE
        
    epsilon : TYPE
        constant
    sigma : TYPE
        Constant
    N : TYPE
        Max length polymer
    wsave : TYPE
        Saved array of weights corresponding to chosen angle per iteration
    Wsave : TYPE
        Sum of possible small weights
    poly_num : TYPE
        Indice that indicates which polymer is being generated
    const : TYPE
        Constant to make sure polymers are not overwritten

    Returns
    -------
    poly_num : TYPE
        Number of a polymer that has been created
    R : TYPE
        Coordinates of a polymer
    wsave : TYPE
        The small weights of the polymer
    Wsave : TYPE
        Sum of the small weights at each step 

    """
    W,w,theta = weights(n, L, R[:,:,poly_num], epsilon, sigma)
    w2 = np.cumsum(w)
    index = roulete(w2)
     
    wsave[L-1,poly_num] = w[int(index)]
    Wsave[L-1,poly_num] = W
    PolWeight = np.cumsum(Wsave[:L,poly_num])[-1]
    
    theta_choice = theta[int(index)]
    R[:,L+1,poly_num] = R[:,L,poly_num]+np.array([np.cos(theta_choice),np.sin(theta_choice)])
    L = L+1
    Uplim = 2 * np.sum(Wsave[:L,poly_num])/Wsave[0,poly_num]/(L-1)
    Lowlim = 1.2 * np.sum(Wsave[:L,poly_num])/Wsave[0,poly_num]/(L-1)
    if int(L)<int(N-1):
        
        if PolWeight>=Uplim and int(L)>int(70):
            
            Wsave[:,poly_num] = Wsave[:,poly_num]*0.5
            Wsave[:,poly_num+const] = Wsave[:,poly_num]*0.5
            R[:,:,poly_num+const] = R[:,:,poly_num]
            simulation_perm(R, L, n, epsilon, sigma,N,wsave,Wsave,poly_num,const+1)
            simulation_perm(R, L, n, epsilon, sigma,N,wsave,Wsave,poly_num+const,const+1)
            
        elif PolWeight<Lowlim and int(L)>int(70):
            Random_num = np.random.uniform(0,1)
            if Random_num<float(0.5):
                Wsave[:,poly_num] = Wsave[:,poly_num]*2
                simulation_perm(R, L, n, epsilon, sigma,N,wsave,Wsave,poly_num,const)
        else:
            simulation_perm(R, L, n, epsilon, sigma,N,wsave,Wsave,poly_num,const)
        
    return poly_num, R , wsave ,Wsave

def run_PERM(R_usefull,W_usefull,iteration_perm,epsilon,sigma,N):
    """
    Parameters
    ----------
    R_usefull :
        Usefull polymer coordinates matrix (empty)
    W_usefull : 
        Weights of the polymers that are used (empty)
    iteration_perm :
        Amount of iterations to run the algorithm
    epsilon :
        Constant
    sigma :
        Constant
    N : 
        Amount of beads

    Returns
    -------
    R_usefull :
        Filtered Coordinates that are used
    W_usefull :
        Filtered weights that are used
    distance_perm :
        Distance between origin and polymer points

    """
    for j in range(iteration_perm):
        size = 500
        R_new,wsave_new,Wsave_new = PERM(size, epsilon, sigma, N)
        
        indices = np.nonzero(R_new[-1,-1,:])
        
        for i in indices:    
            R_usefull = np.append(R_usefull,R_new[:,:,i],2)
            W_usefull = np.append(W_usefull,Wsave_new[:,i],1)
    R_usefull = R_usefull[:,:,1:]
    W_usefull = W_usefull[:,1:]
    distance_perm = np.sqrt(np.sum(R_usefull**2,0))
    
    return R_usefull, W_usefull, distance_perm

def modify_p(P_save,Rpol_avg,N, Group_size, distance_save):
    """
    Parameters
    ----------
    P_save :
        The probability of the polymer occuring        
    Rpol_avg :
        Initialisation
    N : 
        Constant
    Group_size : 
        Constant
    distance_save :
         The end-to-end distance
         
    Returns
    -------
    Rpol_avg :
        The weighted average end-to-end distance. 
    """
    Ptrans = np.zeros((Group_size,N-2))
    Ptrans = P_save.transpose()
    Rpol_avg[0,0] = 0;
    Rpol_avg[1,0] = 1;
    
    for j in range(N-2):
        Rpol_avg[j+2,0] = np.dot(distance_save[j+2,:],Ptrans[:,j])/np.sum(Ptrans[:,j])
    return Rpol_avg

def sigma_p(N,P_save,Group_size,distance_save):
    """
    Parameters
    ----------
    N : 
        Constant
    P_save :
        The probability of the polymer occuring
    Group_size : 
        Constant
    distance_save :
         The squared end-to-end distance
         
    Returns
    -------
    sigma_A :
        The weighted error of the end-to-end distance 
    """
    sigma_A = np.zeros(N)
    sigma_A[0] = 0
    sigma_A[1] = 0
    
    for j in range(N-2):
        sigma_A[j+2] = 1/np.sqrt(np.count_nonzero(P_save[j,:]))*np.sqrt(1/(np.count_nonzero(P_save[j,:])-1)*np.sum((distance_save[j+2,:]- np.dot(distance_save[j+2,:],P_save[j,:])/np.sum(P_save[j,:]))**2))

    return sigma_A