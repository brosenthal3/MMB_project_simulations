import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

class PhaseField2DModel:
    def __init__(self, plotting=True, cancer_division=0.004, cancer_ECM_adhesion=0.15):
        self.Lx = 150
        self.Ly = 200
        self.types = [0, 2, 2, 1, 2] # cell types, 0:substrate, 2:healthy, 1:cancer

        # model parameters
        self.kappa = 22 # interface energy coefficient
        self.epsilon2 = 0.8 # interface thickness parameter
        self.alpha = 0.004 # volume conservation parameter
        self.mobility = [3, 4, 4] # mobility coefficient per cell type
        self.tau = [1, 0.25, 0.3] # relaxation time per cell type
        self.gamma = np.array([[8, 1.5, 30],
                              [1.5, 30, 30],
                              [30, 30, 30]]) # no-overlap coefficient

        self.adhesion = np.array([[0.001, cancer_ECM_adhesion, 0.05],
                                 [cancer_ECM_adhesion, 0.02, 0.01],
                                 [0.05, 0.01, 0.4]]) # adhesion matrix between cell types

        self.tension_strength = [0.002, 1, 1] # strengths for each cell type

        # cell division and death parameters
        self.max_cells = 15
        self.division_rate = [0, cancer_division, 0.0004] # probability of cell division per time step per type
        self.death_line = 140
        self.ECM_line = 90

        # Lists to shift rows and columns by one in the 4 directions
        self.sright = [(i+1)%self.Lx for i in range(self.Lx)] 
        self.sleft = [(i-1)%self.Lx for i in range(self.Lx)] 
        self.sup = [(i+1)%self.Ly for i in range(self.Ly)] 
        self.sdown = [(i-1)%self.Ly for i in range(self.Ly)] 

        self.phi = self.set_phi() # Initial Condition of lattice
        
        self.plotting = plotting
        self.draw_lines = True

    
    def get_params(self):
        """Return a dictionary of model parameters for reporting."""
        params = {
            "Lx": self.Lx,
            "Ly": self.Ly,
            "types": self.types,
            "kappa": self.kappa,
            "epsilon2": self.epsilon2,
            "alpha": self.alpha,
            "mobility": self.mobility,
            "tau": self.tau,
            "gamma": self.gamma.tolist(),
            "adhesion": self.adhesion.tolist(),
            "tension_strength": self.tension_strength,
            "max_cells": self.max_cells,
            "division_rate": self.division_rate,
            "death_line": self.death_line,
            "ECM_line": self.ECM_line
        }
        return params


    def paint_string(self, mask, x0, y0, angle, length, radius):
        """Draws an ECM-like fiber"""

        # Sample points along the line and stamp a disk (radius) at each point
        steps = max(8, int(length))
        xs = x0 + np.linspace(0, length, steps) * np.cos(angle)
        ys = y0 + np.linspace(0, length, steps) * np.sin(angle)
        X, Y = np.ogrid[:self.Lx, :self.Ly]
        r2 = radius * radius
        for cx, cy in zip(xs, ys):
            dist2 = (X - cx) ** 2 + (Y - cy) ** 2
            mask[dist2 <= r2] = 1
        return mask


    def set_phi(self):
        """Set initial condition of the grid with ECM strings and cells"""

        phi = np.zeros((self.Lx,self.Ly, len(self.types)))
        # ECM strings parameters
        rng = np.random.default_rng(0)  # fix seed if you want reproducible patterns
        n_strings = 80
        min_len, max_len = 6, 13
        radius = 1.2
        # paint ECM strings in substrate (k=0)
        for _ in range(n_strings):
            x0 = rng.uniform(0, self.Lx)
            y0 = rng.uniform(0, self.ECM_line)
            angle = rng.uniform(0, 2 * np.pi)
            length = rng.uniform(min_len, max_len)
            phi[:, :, 0] = self.paint_string(phi[:, :, 0], x0, y0, angle, length, radius)

        # cell spacing parameters
        spacing = -40
        d_spacing = 30
        # paint cells (k=1..N)
        for k in range(1, len(self.types)):
            for i in range(self.Lx):
                for j in range(self.Ly):
                    if (i-(self.Lx/2 + spacing))**2 + (j-self.Ly/2)**2 < 10**2:
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
        """Get the middle x-coordinate of a cell based on phi values."""

        indices = np.argwhere(phi > 0.5)
        if indices.size == 0:
            return self.Lx // 2  # default to center if no points found
        mid_x = int(np.mean(indices[:, 0]))
        return mid_x


    def check_cell_death(self, k):
        """Check if the majority of a cell is above the death line."""
        phi_above = np.sum(self.phi[:, self.death_line:, k])
        phi_below = np.sum(self.phi[:, :self.death_line, k])
        if phi_above > phi_below:
            return True
        return False
    
    
    def divide_cell(self, k, VolT):
        VolT[k] /= 1.5  # reset target volume after division
        # perform cell division
        new_phi_left = np.copy(self.phi[:, :, k])
        new_cell_right = np.copy(self.phi[:, :, k])
        mid_x = self.get_middle_of_cell(new_phi_left)
        new_phi_left[:mid_x, :] = 0  # clear half of the original cell
        self.phi[:, :, k] = new_phi_left

        # create new cell
        new_cell_right[mid_x:, :] = 0  # clear other half for new cell
        self.phi = np.concatenate((self.phi, new_cell_right[:, :, np.newaxis]), axis=2)
        self.types.append(self.types[k])  # same type as parent cell
        VolT.append(VolT[k])  # same target volume as parent cell


    def run_simulation(self, tmax=300, dt=0.05):
        # time simulation
        init_time = time.time()

        if self.plotting:
            # Make Plot - use imshow for much faster rendering
            fig, ax = plt.subplots()
            plot_field = np.sum(self.phi, axis=2)
            im = ax.imshow(plot_field.T, vmin=0, vmax=3, origin='lower', cmap='inferno', interpolation='nearest')
            plt.colorbar(im)
            if self.draw_lines:
                ax.axhline(y=self.death_line, color='pink', linestyle='--', alpha=0.4)
                #ax.axhline(y=self.Ly // 2, color='pink', linestyle='--', alpha=0.4)
                ax.axhline(y=self.ECM_line, color='pink', linestyle='--', alpha=0.4)
                # Add labeled y-ticks at the line positions
                ax.set_yticks([self.death_line, self.ECM_line])
                ax.set_yticklabels(['Death line', 'ECM line'])

            plt.ion()  # Turn on interactive mode
            plt.show()

        # compute target volumes at t=0 (cells only, exclude substrate k=0)
        VolT = [np.sum(self.h(self.phi[:, :, k])) + 40 for k in range(len(self.types))]

        # Pre-allocate arrays to avoid repeated allocations
        interface = np.zeros((self.Lx, self.Ly))
        h_phi = np.zeros((self.Lx, self.Ly, len(self.types)))
        
        # Update of the matrix phi
        t=0
        cell_dividing = None
        cell_dead = None
        cell_death_log = []
        pbar = tqdm(total=tmax, desc="Simulating", unit="sec")
        while t<tmax:
            if cell_dead is not None:
                # log cell death
                cell_death_log.append((self.types[cell_dead], t))
                # remove dead cell from simulation
                self.phi = np.delete(self.phi, cell_dead, axis=2)
                self.types.pop(cell_dead)
                VolT.pop(cell_dead)
                # resize h_phi to match new number of cells
                h_phi = np.zeros((self.Lx, self.Ly, len(self.types)))
                cell_dead = None

            # PRE-COMPUTE h(phi) for all cells once per timestep
            for k in range(len(self.types)):
                self.h(self.phi[:, :, k], out=h_phi[:, :, k])

            # update substrate (k=0) and all cells (k=1..N)
            for k in range(0, len(self.types)):
                Vol = np.sum(h_phi[:, :, k])
                np.multiply(self.phi[:,:,k], (1-self.phi[:,:,k]), out=interface)

                # check if cell dies
                if k!= 0 and self.check_cell_death(k):
                    VolT[k] = 0  # set target volume to zero

                # if cell volume is very small, remove it from simulation
                if Vol < 2 and k != 0:
                    cell_dead = k

                # random chance for a cell to divide!
                if (np.random.rand() < self.division_rate[self.types[k]] and cell_dividing is None and Vol >= 0.9 * VolT[k - 1] and len(self.types) < self.max_cells):
                    cell_dividing = k
                    VolT[k] *= 1.5  # increase target volume for division

                # loop over all cells to calculate adhesion/repulsion terms
                k_cell_type = self.types[k]
                adhesion_repulsion_term = np.zeros((self.Lx, self.Ly))
                considered_combinations = []
                for l in range(0, len(self.types)):
                    if l != k and {k, l} not in considered_combinations:
                        l_cell_type = self.types[l]
                        h_phi_l = h_phi[:, :, l] # precompute h(phi_l)
                        adhesion_repulsion_term += interface * ((6 * self.gamma[k_cell_type, l_cell_type]) / self.kappa) * h_phi_l
                        adhesion_repulsion_term -= (6 * self.adhesion[k_cell_type, l_cell_type] / self.kappa) * self.nabla_squared(h_phi_l)
                        considered_combinations.append({k, l})

                # calculate volume exclusion term
                volume_exclusion_term = 6 * self.mobility[self.types[k]] * self.alpha * interface * (VolT[k] - Vol)
                # compute surface tension term
                surface_tension_term = 0.5 * interface * (1-2*self.phi[:,:,k]) - self.epsilon2 * self.nabla_squared(self.phi[:,:,k])
                surface_tension_term *= self.tension_strength[self.types[k]]

                # update phi using the PDE
                self.phi[:, :, k] = self.phi[:, :, k] + dt * ((-1 / self.tau[self.types[k]]) * (surface_tension_term + adhesion_repulsion_term) + volume_exclusion_term)
                # clamp phi values between 0 and 1 
                self.phi[:,:,k] = np.clip(self.phi[:,:,k], 0, 1)

                # perform cell division if volume threshold is reached
                if Vol >= 0.99 * VolT[k] and cell_dividing == k:
                    # perform division
                    self.divide_cell(k, VolT)
                    cell_dividing = None
                    # resize h_phi to match new number of cells
                    h_phi = np.zeros((self.Lx, self.Ly, len(self.types)))

            # update time step
            t=t+dt
            pbar.update(dt)

            # Update plot every 100 steps
            if (round(t/dt)%100==0) and self.plotting:
                plot_field = np.sum(self.phi, axis=2) + np.sum([self.phi[:, :, k] for k in range(len(self.types)) if self.types[k] == 1], axis=0) + 3*self.phi[:, :, 0]  # Emphasize type 1 cells and first cell

                im.set_data(plot_field.T)
                plt.draw()
                plt.title(f"Phase Field Simulation at t={round(t,2)}")
                plt.pause(0.001)

        print("Simulation complete.")
        finish_time = time.time()
        print(f"Total simulation time: {round(finish_time - init_time, 2)} seconds")
        pbar.close()

        if self.plotting:
            plt.ioff()  # Turn off interactive mode
            plt.show()

        # return data for analysis
        params_report = self.get_params()
        return self.phi, self.types, cell_death_log, params_report


    def __call__(self):
        return self.run_simulation()


def plot_phi(self, phi, types):
    # Make Plot - use imshow for much faster rendering
    fig, ax = plt.subplots()
    plot_field = np.sum(phi, axis=2)
    im = ax.imshow(plot_field.T, vmin=0, vmax=3, origin='lower', cmap='inferno', interpolation='nearest')

    plt.colorbar(im)
    if self.draw_lines:
        ax.axhline(y=self.death_line, color='pink', linestyle='--', alpha=0.4)
        #ax.axhline(y=self.Ly // 2, color='pink', linestyle='--', alpha=0.4)
        ax.axhline(y=self.ECM_line, color='pink', linestyle='--', alpha=0.4)
        # Add labeled y-ticks at the line positions
        ax.set_yticks([self.death_line, self.ECM_line])
        ax.set_yticklabels(['Death line', 'ECM line'])

    plot_field = np.sum(phi, axis=2) + np.sum([phi[:, :, k] for k in range(len(types)) if types[k] == 1], axis=0) + 3*phi[:, :, 0]  # Emphasize type 1 cells and first cell

    im.set_data(plot_field.T)  # Update data instead of redrawing
    plt.draw()
    plt.title(f"Phase Field Simulation Final State")

    plt.savefig("results/final_state.png")


if __name__ == "__main__":
    model = PhaseField2DModel(plotting=False)
    results = model()
    plot_phi(model, results[0], results[1])
