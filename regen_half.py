"""Regenerate half_tem_faquarson.csv with the corrected time-stepping.

Faithful mirror of forward_farquarson.py (same mesh incl. block refinement, same
volume-averaged model, same source/receiver) with the fixed x1.4 ramp. Uses whatever
direct solver is installed; the linear solution is solver-independent.
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from discretize import TensorMesh
from discretize import utils as dis_utils
from discretize.utils import volume_average
from simpeg import utils, maps
from simpeg.electromagnetics import time_domain
import pandas as pd
from scipy.special import erf
from scipy.constants import mu_0


def get_solver():
    from scipy.sparse import eye
    probe = eye(2).tocsr()
    for name in ("Pardiso", "Mumps"):
        try:
            import pymatsolver
            S = getattr(pymatsolver, name); S(probe); return S
        except Exception:
            continue
    from pymatsolver import SolverLU
    return SolverLU


Solver = get_solver()
print("solver:", Solver.__name__)

# --- geometry / model (identical to forward_farquarson.py) ---
dx = dy = 100.0
v = lambda sx, sy: np.array([sx*dx/2, sy*dy/2, 0.])
source_locations = np.vstack((v(-1, -1), v(-1, 1), v(1, 1), v(1, -1), v(-1, -1)))
rx_locations = np.array([[0., 0., 0.]])

hx = np.ones(40)*5; hy = np.ones(30)*5; hz = np.ones(31)*5
mesh_tensor = TensorMesh([hx, hy, hz], x0=[-hx.sum()/2, -hy.sum()/2, -hz.sum()+5])
p0 = np.array([30, -50, -60]); p1 = np.array([70, 50, -30])
inds_block = utils.model_builder.get_indices_block(p0, p1, mesh_tensor.cell_centers)
blk_pts = mesh_tensor.cell_centers[inds_block, :]
active = mesh_tensor.cell_centers[:, 2] < 0.0
sigma = np.ones(mesh_tensor.n_cells)*0.1
sigma[inds_block] = 0.1
sigma[~active] = 1e-8

# --- tree mesh (identical refinement to forward_farquarson.py) ---
x_buffer = y_buffer = 200
locs = np.vstack((source_locations, rx_locations))
x = np.linspace(locs[:, 0].min()-x_buffer, locs[:, 0].max()+x_buffer)
y = np.linspace(locs[:, 1].min()-y_buffer, locs[:, 1].max()+y_buffer)
dem = utils.ndgrid(x, y, np.r_[0.])
mesh = dis_utils.mesh_builder_xyz(dem, [10, 10, 5], padding_distance=[5000, 5000, 5000],
                                  base_mesh=None, depth_core=100, mesh_type='tree')
mesh.refine_surface(dem, -1, padding_cells_by_level=[[0, 0, 1], [0, 0, 15]], finalize=False)
mesh.refine_line(source_locations, -1, finalize=False)
mesh.refine_points(rx_locations, -1, finalize=False)
mesh.refine_points(blk_pts, -1, padding_cells_by_level=[1, 0, 0], finalize=False)
mesh.finalize()
print(f"n_cells = {mesh.n_cells/1e3:.1f}k")

sig_tree = volume_average(mesh_tensor, mesh) @ sigma

times = np.logspace(-4, -2, 21)
rx = time_domain.Rx.PointMagneticFluxTimeDerivative(rx_locations, times, "z")
src = time_domain.Src.LineCurrent([rx], location=source_locations)
survey = time_domain.Survey([src])


def generate_time_steps(n_constant_steps, increase_rate, start_time_step, n_per_step):
    return np.hstack([np.ones(n_per_step)*start_time_step*increase_rate**i
                      for i in range(n_constant_steps)])


# FIXED time-stepping (matches the edited forward_farquarson.py)
time_steps = generate_time_steps(26, 1.4, 3e-7, 6)
sim = time_domain.Simulation3DElectricField(
    mesh, sigmaMap=maps.IdentityMap(nP=mesh.n_cells), survey=survey,
    time_steps=time_steps, solver=Solver)
dpred = sim.dpred(sig_tree)

# analytic check
a = np.sqrt(1e4/np.pi); theta = np.sqrt(mu_0*0.1/(4*times)); ta = theta*a
anal = 1/(0.1*a**3)*(3*erf(ta)-2/np.sqrt(np.pi)*ta*(3+2*ta**2)*np.exp(-ta**2))
ratio = np.abs(dpred)/np.abs(anal)
print(f"vs analytic: mean={ratio.mean():.3f}  range={ratio.min():.3f}-{ratio.max():.3f}")

pd.DataFrame(data=np.c_[times, dpred], columns=['times_s', 'voltage_V_Am2']).to_csv(
    'half_tem_faquarson.csv', index=False)
print("wrote half_tem_faquarson.csv")
