import numpy as np
import matplotlib.pyplot as plt 

# Parameters
epsilon2=1

# Initial Condition (random values around 0.5)
Lx=100
Ly=100
phi = np.ones((Lx,Ly))*0.5+0.1*(np.random.rand(Lx,Ly)-0.5)

# Make Plot
fig = plt.figure()
plt.pcolor(phi)
plt.colorbar(ticks=[0,0.5,1])

# Simulation time parameters
tmax=100
dt=0.1
t=0

# Lists to shift rows and columns by one in the 4 directions
sright = [(i+1)%Lx for i in range(Lx)] 
sleft = [(i-1)%Lx for i in range(Lx)] 
sup = [(i+1)%Ly for i in range(Ly)] 
sdown = [(i-1)%Ly for i in range(Ly)] 

# Update of the matrix phi
while t<tmax:
    phi = phi + dt * (- 0.5*(1-phi)*phi*(1-2*phi) + epsilon2* ( phi[sright,:] + phi[sleft,:] + phi[:,sup] + phi[:,sdown] - 4*phi ) ) 
    t=t+dt
        
    if (round(t/dt)%100==0):
        plt.cla()
        plt.pcolor(phi, vmin=0, vmax=1)
        plt.pause(0.001)

plt.show()