import numpy as np
import warnings, time as timemod
warnings.filterwarnings("ignore")
from discretize import TensorMesh
from discretize import utils as dis_utils
from discretize.utils import volume_average
from simpeg import utils, maps
from simpeg.electromagnetics import time_domain
from pymatsolver import Mumps as Pardiso
from scipy.special import erf
from scipy.constants import mu_0

# ---------- geometry / model (mirrors forward_farquarson.py) ----------
dx = dy = 100.0
z_surface = 0.0
v1 = np.array([-dx/2,-dy/2,0.]); v2 = np.array([-dx/2,dy/2,0.])
v3 = np.array([dx/2,dy/2,0.]); v4 = np.array([dx/2,-dy/2,0.])
source_locations = np.vstack((v1,v2,v3,v4,v1))
rx_locations = np.array([[0.,0.,0.]])
times = np.logspace(-4,-2,21)
sigma_hs = 0.1

# background tensor model for volume-averaging onto the tree
hx = np.ones(40)*5; hy = np.ones(30)*5; hz = np.ones(31)*5
mesh_tensor = TensorMesh([hx,hy,hz], x0=[-hx.sum()/2,-hy.sum()/2,-hz.sum()+5])
active = mesh_tensor.cell_centers[:,2] < 0.0
sigma_t = np.ones(mesh_tensor.n_cells)*sigma_hs
sigma_t[~active] = 1e-8

def build_mesh(h):
    x_buffer = y_buffer = 200
    locations = np.vstack((np.vstack(source_locations), rx_locations))
    x_core = (locations[:,0].min()-x_buffer, locations[:,0].max()+x_buffer)
    y_core = (locations[:,1].min()-y_buffer, locations[:,1].max()+y_buffer)
    x = np.linspace(*x_core); y = np.linspace(*y_core)
    dem = utils.ndgrid(x, y, np.r_[z_surface])
    mesh = dis_utils.mesh_builder_xyz(dem, h, padding_distance=[5000,5000,5000],
                                      base_mesh=None, depth_core=100, mesh_type='tree')
    mesh.refine_surface(dem, -1, padding_cells_by_level=[[0,0,1],[0,0,15]], finalize=False)
    mesh.refine_line(source_locations, -1, finalize=False)
    mesh.refine_points(rx_locations, -1, finalize=False)
    mesh.finalize()
    return mesh

def gen_steps(n_constant_steps, increase_rate, start_time_step, n_per_step):
    out = []
    for i in range(n_constant_steps):
        out.append(np.ones(n_per_step)*start_time_step*increase_rate**i)
    return np.hstack(out)

def analytic(radius):
    theta = np.sqrt(mu_0*sigma_hs/(4*times)); ta = theta*radius
    return 1/(sigma_hs*radius**3)*(3*erf(ta)-2/np.sqrt(np.pi)*ta*(3+2*ta**2)*np.exp(-ta**2))

a = np.sqrt(1e4/np.pi)
d_anal = analytic(a)

def run(h, steps, label):
    t0 = timemod.time()
    mesh = build_mesh(h)
    P = volume_average(mesh_tensor, mesh)
    sig_tree = P @ sigma_t
    rx = time_domain.Rx.PointMagneticFluxTimeDerivative(rx_locations, times, "z")
    src = time_domain.Src.LineCurrent([rx], location=source_locations)
    survey = time_domain.Survey([src])
    sim = time_domain.Simulation3DElectricField(
        mesh, sigmaMap=maps.IdentityMap(nP=mesh.n_cells), survey=survey,
        time_steps=steps, solver=Pardiso)
    d = sim.dpred(sig_tree)
    dt = timemod.time()-t0
    r = np.abs(d)/np.abs(d_anal)
    print(f"[{label}] cells={mesh.n_cells/1e3:.1f}k nsteps={len(steps)} "
          f"t_max={steps.sum():.4e} | runtime={dt:.0f}s | ratio min={r.min():.3f} "
          f"max={r.max():.3f} mean={r.mean():.3f}", flush=True)
    return d

configs = {
 "baseline":   dict(h=[10,10,5], steps=gen_steps(11,2.0,1e-6,5)),    # original
 "time_fine":  dict(h=[10,10,5], steps=gen_steps(18,1.5,1e-6,5)),    # finer time, orig mesh
 "time_finer": dict(h=[10,10,5], steps=gen_steps(26,1.4,3e-7,6)),    # finest time, orig mesh
 "mesh_fine":  dict(h=[5,5,2.5],  steps=gen_steps(11,2.0,1e-6,5)),   # finer mesh, ORIG time steps
 "mesh+time":  dict(h=[5,5,2.5],  steps=gen_steps(18,1.5,1e-6,5)),   # both refined
}

results = {"analytic": d_anal, "times": times}
for name, cfg in configs.items():
    results[name] = run(cfg["h"], cfg["steps"], name)

np.savez("/tmp/convergence_results.npz", **results)
print("DONE")
