import numpy as np
import matplotlib.pyplot as plt 

def h(x):
    h = x*x*(3-2*x)
    return(h)


# Parameters
epsilon2=1
alpha=0.01


# Initial Condition (random values around 0.5)
Lx=100
Ly=100

phi = np.zeros((Lx,Ly))
for i in range(Lx):
    for j in range(Ly):
        if (i-Lx/2)**2 + (j-Ly/2)**2/4 < 20**2:
            phi[i,j]=1

VolT = np.sum(h(phi))

            
# Make Plot
plt.figure()
plt.pcolor(phi)
plt.colorbar()

# Simulation time parameters
tmax=500
dt=0.1
t=0

# Lists to shift rows and columns by one in the 4 directions
sright = [(i+1)%Lx for i in range(Lx)] 
sleft = [(i-1)%Lx for i in range(Lx)] 
sup = [(i+1)%Ly for i in range(Ly)] 
sdown = [(i-1)%Ly for i in range(Ly)] 


# Update of the matrix phi
while t<tmax:
    
    Vol = np.sum(h(phi))
    phi = phi + dt * (- 0.5*(1-phi)*phi*(1-2*phi) + epsilon2* ( phi[sright,:] + phi[sleft,:] + phi[:,sup] + phi[:,sdown] - 4*phi )  - alpha*phi*(1-phi)*(Vol-VolT) ) 
    
    t=t+dt
        
    if (round(t/dt)%100==0):
        plt.figure()
        plt.pcolor(phi, vmin=0, vmax=1)
        plt.colorbar(ticks=[0,0.5,1])
        plt.pause(0.001)
