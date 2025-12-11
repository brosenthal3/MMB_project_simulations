import numpy as np
import time
import matplotlib.pyplot as plt

class PhaseField2DModel:
    def __init__(self, Lx=150, Ly=100):
        self.Lx = Lx
        self.Ly = Ly
        self.N = 4 # number of cells
        self.types = [2, 2, 1, 2] # cell types

        # model parameters
        self.kappa = 8 # interface energy coefficient
        self.epsilon2 = 0.8 # interface thickness parameter
        self.alpha = 0.01 # volume conservation parameter
        self.mobility = 2 # mobility coefficient
        self.tau = 1 # relaxation time
        self.gamma = 22 # no-overlap coefficient
        self.adhesion = np.array([[0.001, 0.001],
                                 [0.001, 1.5]]) # adhesion matrix between cell types

        # Lists to shift rows and columns by one in the 4 directions
        self.sright = [(i+1)%self.Lx for i in range(self.Lx)] 
        self.sleft = [(i-1)%self.Lx for i in range(self.Lx)] 
        self.sup = [(i+1)%self.Ly for i in range(self.Ly)] 
        self.sdown = [(i-1)%self.Ly for i in range(self.Ly)] 

        self.phi = self.set_phi() # Initial Condition of lattice


    def set_phi(self):
        phi = np.zeros((self.Lx,self.Ly, self.N))
        spacing = -40  # horizontal spacing between cell centers
        d_spacing = 25  # increment of spacing for each cell
        
        for k in range(self.N):
            for i in range(self.Lx):
                for j in range(self.Ly):
                    if (i-(self.Lx/2 + spacing))**2 + (j-self.Ly/2)**2 < 13**2:
                        phi[i, j, k] = 1

            spacing += d_spacing

        return phi


    def h(self, x, out=None):
        """Smooth step polynomial h(x) with optional output buffer to reduce allocations."""
        if out is None:
            return x * x * (3 - 2 * x)
        # In-place: out = x*x; then out *= (3 - 2*x)
        np.multiply(x, x, out=out)
        np.multiply(out, (3 - 2 * x), out=out)
        return out
    

    def nabla_squared(self, phi):
        """Laplacian via roll; uses wraparound boundaries and avoids indexing lists."""
        return (
            np.roll(phi, 1, axis=0)
            + np.roll(phi, -1, axis=0)
            + np.roll(phi, 1, axis=1)
            + np.roll(phi, -1, axis=1)
            - 4 * phi
        )


    def run_simulation(self, tmax=250, dt=0.025):
        # time simulation
        init_time = time.time()

        # Make Plot - use imshow for much faster rendering
        fig, ax = plt.subplots()
        plot_field = np.sum(self.phi, axis=2)
        im = ax.imshow(plot_field.T, vmin=0, vmax=2, origin='lower', cmap='inferno', interpolation='nearest')
        plt.colorbar(im)
        plt.ion()  # Turn on interactive mode
        plt.show()

        # compute volume at t=0
        VolT = [np.sum(self.h(self.phi[:, :, k])) for k in range(self.N)]
        
        # Update of the matrix phi
        t=0
        considered_combinations = []
        while t<tmax:
            for k in range(self.N):
                Vol = np.sum(self.h(self.phi[:, :, k]))
                interface = self.phi[:,:,k] * (1-self.phi[:,:,k])

                # compute surface tension term
                surface_tension_term = 0.5 * interface * (1-2*self.phi[:,:,k]) - self.epsilon2 * self.nabla_squared(self.phi[:,:,k])

                # loop over all cells to calculate adhesion/repulsion terms
                adhesion_repulsion_term = 0
                for l in range(self.N):
                    if l != k and {k,l} not in considered_combinations:
                        k_cell_type = self.types[k]-1
                        l_cell_type = self.types[l]-1
                        adhesion_repulsion_term += interface * ( (6*self.gamma)/self.kappa) * self.h(self.phi[:,:,l]) - (6*self.adhesion[k_cell_type, l_cell_type]/self.kappa * self.nabla_squared(self.h(self.phi[:,:,l])))
                        considered_combinations.append({k,l})
    
                # calculate volume exclusion term
                volume_exclusion_term = 6 * self.mobility * self.alpha * interface * (VolT[k]-Vol)

                # update phi using the PDE
                self.phi[:,:,k] = self.phi[:,:,k] + dt*((-1/self.tau) * (surface_tension_term + adhesion_repulsion_term) + volume_exclusion_term)
                # clamp phi values between 0 and 1 
                self.phi[:,:,k] = np.clip(self.phi[:,:,k], 0, 1)

            # update time step
            t=t+dt

            # Update plot every 100 steps
            if (round(t/dt)%100==0):
                plot_field = np.sum(self.phi, axis=2) + self.phi[:, :, 2]

                im.set_data(plot_field.T)  # Update data instead of redrawing
                plt.draw()
                plt.pause(0.001)  # Small pause for display update

        print("Simulation complete.")
        finish_time = time.time()
        print(f"Total simulation time: {round(finish_time - init_time, 2)} seconds")

        plt.ioff()  # Turn off interactive mode
        plt.show()


    def __call__(self):
        self.run_simulation()


if __name__ == "__main__":
    model = PhaseField2DModel()
    model()