
import numpy as np
import matplotlib.pyplot as plt 


#___________________________________________________________________________

# Function to make file for pretty plots
def save_vti_file(phi, nx, ny, nz, name):
    
    valor=[]
    for iz in range(nz-1):
        for iy in range(ny-1):
            for ix in range(nx-1):
                valor.append(phi[ix,iy,iz])
    
    pc_string_novo = "    ".join([str(_) for _ in valor]) # criacao de uma string com os valores da lista
    
    with open(name + ".vti", "w" ) as my_file:
        my_file.write('<?xml version="1.0"?>')
        my_file.write('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
        my_file.write('  <ImageData WholeExtent="0 '+str(nx-1)+' 0 '+str(ny-1)+' 0 '+str(nz-1)+'" Origin="0 0 0" Spacing ="1 1 1">\n')
        my_file.write('    <Piece Extent="0 '+str(nx-1)+' 0 '+str(ny-1)+' 0 '+str(nz-1)+'">\n') # dimensao da matriz x1 x2 y1 y2 z1 z2
        my_file.write('     <CellData>\n')
        my_file.write('     <DataArray Name="scalar_data" type="Float64" format="ascii">\n')
        my_file.write('     ')
        my_file.write(pc_string_novo)
        my_file.write('\n         </DataArray>\n')
        my_file.write('      </CellData>\n')
        my_file.write('    </Piece>\n')
        my_file.write('</ImageData>\n')
        my_file.write('</VTKFile>\n')
        my_file.close() # fecha o ficheiro

#___________________________________________________________________________



# Parameters
epsilon2=1


# Initial Condition (random values around 0.5)
Lx=50
Ly=50
Lz=50
phi = np.ones((Lx,Ly,Lz))*0.5+0.1*(np.random.rand(Lx,Ly,Lz)-0.5)

# Make Plot (slice at middle of box)
plt.figure()
plt.pcolor(phi[:,:,Lz//2])
plt.colorbar()

# Save file for 3D plot
nome = "PF_res_0"
save_vti_file(phi,Lx,Ly,Lz,nome)

# Simulation time parameters
tmax=100
dt=0.1
t=0

# Lists to shift rows and columns by one in the 6 directions
sright = [(i+1)%Lx for i in range(Lx)] 
sleft = [(i-1)%Lx for i in range(Lx)] 
sup = [(i+1)%Ly for i in range(Ly)] 
sdown = [(i-1)%Ly for i in range(Ly)] 
sfront = [(i+1)%Lz for i in range(Lz)] 
sback = [(i-1)%Lz for i in range(Lz)] 



# Update of the matrix phi
while t<tmax:
    
    phi = phi + dt * (- 0.5*(1-phi)*phi*(1-2*phi) + epsilon2* ( phi[sright,:,:] + phi[sleft,:,:] + phi[:,sup,:] + phi[:,sdown,:] + phi[:,:,sfront] + phi[:,:,sback] - 6*phi ) ) 
    
    t=t+dt
        
    if (round(t/dt)%100==0):
        plt.figure()
        plt.pcolor(phi[:,:,Lz//2], vmin=0, vmax=1)
        plt.colorbar(ticks=[0,0.5,1])
        plt.pause(0.001)
        nome = "PF_res_" + str(round(t/dt)) 
        save_vti_file(phi,Lx,Ly,Lz,nome)




