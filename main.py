

#TODO run simulation deterministc
#TODO rename
#TODO change powers to log 
#TODO named tuple

import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from scipy.stats import truncnorm

def V(y, eps):
    return (eps*y**2)/2 + (y**4)/4

def dV_dy(y: ndarray, eps: ndarray):
    return (eps*y) + y**3
    
def du_det(u: ndarray, eps: ndarray):
    u1, u2 = u[:, 0], u[:, 1]  
    du1_dt = u2
    du2_dt = -2 * u2 - dV_dy(u1, eps)
    du = np.column_stack((du1_dt, du2_dt))
    return du

def sim_det(seed:int , T: int, dt: int, N: int):
    time = np.arange(0, T, dt)  # Time axis
    us = np.empty((N, len(time), 2)) # Emtpy Array
    us[:, 0, :] = [1, 1] # Init u(0)
    eps = sample_eps(seed, N) 

    assert eps.shape[0] == N

    for i in range(1, len(time)):
        dt = time[i] - time[i-1]
        u = us[:, i-1, :] + du_det(us[:, i-1], eps) * dt
        us[:, i, :] = u
    return us

def sample_eps(seed, N):
    random_state = np.random.RandomState(seed)
    # Sample from the truncated normal distribution
    samples = truncnorm.rvs(5, np.inf, loc=0, scale=1, size=N, random_state=random_state)
    return samples

def plot_trajectory(us):
    for u in us:
        x, y = u.T
        t = np.linspace(0, 1, len(x))

        plt.scatter(x, y, c=t, cmap='viridis', s=10)  # Use 'viridis' or any other colormap
        plt.xlabel("u1")
        plt.ylabel("u2")
        plt.title(f"Trajectory of {us.shape[0]} samples")
    plt.show()

if __name__ == "__main__":
    N = 100
    T = 20
    dt = 0.1
    seed = np.arange(0, N)

    plot_trajectory(
        sim_det(seed, T, dt, N)
    )
