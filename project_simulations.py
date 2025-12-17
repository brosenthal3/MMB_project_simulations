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
        self.kappa = 25 # interface energy coefficient
        self.epsilon2 = 0.9 # interface thickness parameter
        self.alpha = 0.005 # volume conservation parameter
        self.mobility = 4 # mobility coefficient
        self.tau = 0.2 # relaxation time
        self.gamma = 43 # no-overlap coefficient
        self.adhesion = np.array([[0.01, 0.01],
                                 [0.01, 0.05]]) # adhesion matrix between cell types
        self.max_cells = 10
        self.division_rate = 0.0003 # probability of cell division per time step

        # Lists to shift rows and columns by one in the 4 directions
        self.sright = [(i+1)%self.Lx for i in range(self.Lx)] 
        self.sleft = [(i-1)%self.Lx for i in range(self.Lx)] 
        self.sup = [(i+1)%self.Ly for i in range(self.Ly)] 
        self.sdown = [(i-1)%self.Ly for i in range(self.Ly)] 

        self.phi = self.set_phi() # Initial Condition of lattice


    def set_phi(self):
        phi = np.zeros((self.Lx,self.Ly, self.N))
        spacing = -40  # horizontal spacing between cell centers
        d_spacing = 28  # increment of spacing for each cell
        
        for k in range(self.N):
            for i in range(self.Lx):
                for j in range(self.Ly):
                    if (i-(self.Lx/2 + spacing))**2 + (j-self.Ly/2)**2 < 12**2:
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
    
    def get_middle_of_cell(self, phi):
        indices = np.argwhere(phi > 0.5)
        if indices.size == 0:
            return self.Lx // 2  # default to center if no points found
        mid_x = int(np.mean(indices[:, 0]))
        return mid_x


    def run_simulation(self, tmax=200, dt=0.045):
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
        VolT = [np.sum(self.h(self.phi[:, :, k]))+50 for k in range(self.N)]
        
        # Update of the matrix phi
        t=0
        considered_combinations = []
        cell_dividing = None
        while t<tmax:
            for k in range(self.N):

                Vol = np.sum(self.h(self.phi[:, :, k]))
                interface = self.phi[:,:,k] * (1-self.phi[:,:,k])
                # if (round(t/dt)%100==0):
                #     print(f"Volume of cell {k}: {Vol}, Target Volume: {VolT[k]}")

                # random chance for a cell to divide!
                if np.random.rand() < self.division_rate and cell_dividing is None and Vol >= 0.9*VolT[k] and self.N <= self.max_cells:
                    print(f"Cell {k} is growing")
                    cell_dividing = k
                    VolT[k] *= 1.5  # increase target volume for division


                if Vol >= 0.99*VolT[k] and cell_dividing == k:
                    VolT[k] /= 1.5  # reset target volume after division
                    # perform cell division
                    new_phi_left = np.copy(self.phi[:, :, k])
                    new_cell_right = np.copy(self.phi[:, :, k])
                    mid_x = self.get_middle_of_cell(new_phi_left)
                    new_phi_left[:mid_x, :] = 0  # clear half of the original cell
                    self.phi[:, :, k] = new_phi_left

                    # create new cell)
                    new_cell_right[mid_x:, :] = 0  # clear other half for new cell
                    self.phi = np.concatenate((self.phi, new_cell_right[:, :, np.newaxis]), axis=2)
                    self.N += 1
                    self.types.append(self.types[k])  # same type as parent cell
                    VolT.append(VolT[k])  # same target volume as parent cell
                    cell_dividing = None  # reset dividing cell
                    
                    print(f"Cell {k} of type {self.types[k]} is dividing at time {round(t,2)}, new type list: {self.types}")

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
                
                considered_combinations = []

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

                plot_field = np.sum(self.phi, axis=2) +np.sum([self.phi[:, :, k] for k in range(self.N) if self.types[k] == 1], axis=0)

                im.set_data(plot_field.T)  # Update data instead of redrawing
                plt.draw()
                plt.pause(0.001)  # Small pause for display update

        print(self.phi.shape[2], " Cells")
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