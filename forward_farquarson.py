#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# Set up a 3D octree mesh
from discretize import TreeMesh
import matplotlib.pyplot as plt
from simpeg import utils
from discretize import utils as dis_utils

import warnings
import os
# Ignore all warnings
warnings.filterwarnings("ignore")


# In[2]:


dx = 100. 
dy = 100. 
z_surface = 0.
dx1 = np.array([-dx/2, -dy/2, 0.])
dx2 = np.array([-dx/2, +dy/2, 0.])
dx3 = np.array([+dx/2, +dy/2, 0.])
dx4 = np.array([+dx/2, -dy/2, 0.])
source_locations = np.vstack((dx1, dx2, dx3, dx4, dx1))
rx_locations = np.array([[0., 0., 0.]])


# In[3]:


plt.plot(source_locations[:,0], source_locations[:,1])
plt.plot(rx_locations[:,0], rx_locations[:,1], 'x')


# In[4]:


from discretize import TensorMesh


# In[5]:


hx = np.ones(40) * 5
hy = np.ones(30) * 5
hz = np.ones(31) * 5
mesh_tensor = TensorMesh([hx, hy, hz], x0=[-hx.sum()/2, -hy.sum()/2, -hz.sum()+5])


# In[6]:


p0 = np.array([30, -50, -60])
p1 = np.array([70, 50, -30])
inds_block = utils.model_builder.get_indices_block(p0, p1, mesh_tensor.cell_centers)


# In[7]:


blk_pts = mesh_tensor.cell_centers[inds_block,:]


# In[8]:


active = mesh_tensor.cell_centers[:,2] < 0.0
sigma = np.ones(mesh_tensor.n_cells) * 0.1
sigma[inds_block] = 0.1
sigma[~active] = 1e-8


# In[9]:


from matplotlib.colors import LogNorm


# In[10]:


mesh_tensor.plot_3d_slicer(sigma, zslice=-35, pcolor_opts={'norm': LogNorm()})


# In[11]:


# Setting up a core lateral extent
x_buffer = 200
y_buffer = 200
locations = np.vstack((np.vstack(source_locations), rx_locations))
xmin = locations[:,0].min(axis=0)
ymin = locations[:,1].min(axis=0)
xmax = locations[:,0].max(axis=0)
ymax = locations[:,1].max(axis=0)
x_core_lim = (xmin-x_buffer, xmax+x_buffer)
y_core_lim = (ymin-y_buffer, ymax+y_buffer)


# In[12]:


x = np.linspace(x_core_lim[0], x_core_lim[1])
y = np.linspace(y_core_lim[0], y_core_lim[1])
dem = utils.ndgrid(x, y, np.r_[z_surface])
h = [10, 10, 5]
mesh = dis_utils.mesh_builder_xyz(
    dem,
    h,
    padding_distance=[5000, 5000, 5000],
    base_mesh=None,
    depth_core=100,
    mesh_type='tree'
)
# here -1 is the maximum octree level; padding_cells_by_level determines expansion in each direction [x, y, z]
mesh.refine_surface(dem, -1, padding_cells_by_level=[[0, 0, 1], [0, 0, 15]], finalize=False)
# here 11 means the maximum octree level, which you can access with mesh.max_level
# Not needed because we refined the surface where the source is located
mesh.refine_line(source_locations, -1, finalize=False)
mesh.refine_points(rx_locations, -1, finalize=False)
mesh.refine_points(blk_pts, -1, padding_cells_by_level=[1, 0, 0], finalize=False)
mesh.finalize()
print (f" the number of cells is {mesh.n_cells/1e3}k")

# In[13]:


from geoana import em
em.tdem.diffusion_distance(1e-4, 2)/4


# In[14]:


from discretize.utils import volume_average


# In[15]:


P = volume_average(mesh_tensor, mesh)


# In[16]:


sig_model_tree = P @ sigma


# In[17]:


# mesh_tensor.write_vtk("test_3d.vtr", models={'sigma':sig_model})


# In[18]:


fig, ax = plt.subplots(1,1, figsize=(10, 5))
out = mesh.plot_slice(
    np.log10(sig_model_tree), normal='Y', ax=ax, grid=True, pcolor_opts={'cmap':'turbo'}, 
    # ind=521
)
plt.colorbar(out[0])
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_aspect(1)


# In[19]:


times = np.logspace(-4, -2, 21)


# In[20]:


area = 100*100
radius = np.sqrt(area/np.pi)


# In[21]:


from simpeg.electromagnetics import time_domain
from pymatsolver import Pardiso
from simpeg import (
    discretize,
    maps,
)

rx_dbdt_z = time_domain.Rx.PointMagneticFluxTimeDerivative(
    rx_locations, times, "z"
)


source_list = []

src = time_domain.Src.LineCurrent([rx_dbdt_z], location=source_locations)
#src = time_domain.Src.CircularLoop([rx_dbdt_z], radius=radius, location=np.array([0., 0., 0.]))
source_list.append(src)
survey = time_domain.Survey(source_list)    
#time_steps = [(3e-6, 8), (1e-5, 8), (3e-5, 8), (1e-4, 8), (3e-4, 8), (1e-3, 7)]

import numpy as np
def generate_time_steps(
    n_constant_steps=15, increase_rate=3, start_time_step=5e-7, n_per_step=5
):
    time_steps = []
    for i_step in range(n_constant_steps):
        ht = np.ones(n_per_step) * start_time_step*increase_rate**i_step
        time_steps.append(ht)
    print("t_min:{:.1e}, t_max:{:.1e}".format(start_time_step, np.sum(time_steps)))
    return np.hstack(time_steps)

# NOTE: the original coarse x2 ramp (increase_rate=2, 55 steps) under-resolves the
# backward-Euler decay and biased dbz/dt high by +7-24% vs the analytic half-space.
# The gentler x1.4 ramp below (156 steps) brings the 3D response to within ~2% of
# analytic on the same mesh. See report.pdf / halfspace_validation_and_timestep_fix.ipynb.
# old: generate_time_steps(n_constant_steps=11, increase_rate=2,   start_time_step=1e-6, n_per_step=5)
time_steps = generate_time_steps(n_constant_steps=26, increase_rate=1.4, start_time_step=3e-7, n_per_step=6)


simulation = time_domain.Simulation3DElectricField(
    mesh, 
    sigmaMap=maps.IdentityMap(nP=mesh.n_cells), 
    survey=survey, 
    time_steps=time_steps,
    solver=Pardiso, 
    verbose=False # print interim results
)


# In[22]:


plt.semilogx(times, np.ones_like(times), '.')
plt.semilogx(simulation.times, np.ones_like(simulation.times), '|', ms=10)


# In[ ]:


dpred = simulation.dpred(sig_model_tree)


# In[ ]:


n_time = len(times)


# In[ ]:


from simpeg import Data


# In[ ]:


# digitized_data = np.array([
#     [8.785674e-05, 2.905499e-05],
#     [1.107074e-04, 2.258795e-05],
#     [1.356845e-04, 1.245740e-05],
#     [1.602578e-04, 8.441922e-06],
#     [2.000805e-04, 5.591327e-06],
#     [2.497988e-04, 3.789037e-06],
#     [3.236236e-04, 2.452799e-06],
#     [3.929864e-04, 1.516752e-06],
#     [4.684710e-04, 1.051644e-06],
#     [6.239944e-04, 5.934168e-07],
#     [7.935940e-04, 3.348505e-07],
#     [1.009290e-03, 1.846722e-07],
#     [1.248491e-03, 1.018479e-07],
#     [1.617467e-03, 5.125563e-08],
#     [2.095488e-03, 3.097802e-08],
#     [2.765458e-03, 1.872258e-08],
#     [3.517100e-03, 9.640408e-09],
#     [4.514592e-03, 5.078850e-09],
#     [5.741642e-03, 2.555964e-09],
#     [7.370041e-03, 1.346557e-09],
# ])


# In[ ]:


fig = plt.figure(figsize=(5,5))
plt.loglog(times, dpred, '--')
plt.grid(True, which='both', alpha=0.5)
plt.xlim(6e-5, 1e-2)
plt.ylim(3e-10, 1e-4)
plt.xlabel("Time (s)")
plt.ylabel("dbz/dt (V/A-m$^2$)")


# In[ ]:


import pandas as pd
df = pd.DataFrame(data=np.c_[times, dpred], columns=['times_s', 'voltage_V_Am2'])
df.to_csv('half_tem_faquarson.csv', index=False)

print ("TEM simulation done")
# In[ ]:




