"""Engine for the 'cheapest 3D square-loop simulation within ~3% of SimPEG 1D' study.

Cost model: with a direct solver the cost is dominated by matrix factorization, done once per
DISTINCT time-step size. So cost ~ (n_distinct_dt x factor(n_cells)) + (n_steps x backsolve(n_cells)).
We measure wall-clock of sim.dpred() as the primary metric and log n_cells / n_distinct_dt / n_steps.

Accuracy metric: max |dBz/dt(3D) / dBz/dt(1D square) - 1| over the 21 channels (also report mean).

Results are cached to explore_cache.npz keyed by the full config, so re-runs are free and the
companion notebook reproduces from cache."""
import numpy as np, time as _time, os, hashlib, warnings
warnings.filterwarnings("ignore")
from scipy.special import erf
from scipy.constants import mu_0
from discretize import utils as dis_utils
from discretize import TensorMesh
from discretize.utils import volume_average
from simpeg import utils, maps
from simpeg.electromagnetics import time_domain
import simpeg.electromagnetics.time_domain as tdem


def get_solver():
    from scipy.sparse import eye
    probe = eye(2).tocsr()
    for name in ("Pardiso", "Mumps"):
        try:
            import pymatsolver; S = getattr(pymatsolver, name); S(probe); return S
        except Exception:
            continue
    from pymatsolver import SolverLU; return SolverLU
Solver = get_solver()

# ---- survey / model ---------------------------------------------------------------------------
dx = dy = 100.0
source_locations = np.array([[-dx/2, -dy/2, 0.], [-dx/2, dy/2, 0.], [dx/2, dy/2, 0.],
                             [dx/2, -dy/2, 0.], [-dx/2, -dy/2, 0.]])
rx_locations = np.array([[0., 0., 0.]])
times = np.logspace(-4, -2, 21)
sigma_hs = 0.1
a = np.sqrt(dx*dy/np.pi)


def _analytic(t):
    theta = np.sqrt(mu_0*sigma_hs/(4*t)); ta = theta*a
    return np.abs(1/(sigma_hs*a**3)*(3*erf(ta) - 2/np.sqrt(np.pi)*ta*(3+2*ta**2)*np.exp(-ta**2)))
d_analytic = _analytic(times)


def run_1d_square(n_points_per_path=10):
    rx = tdem.receivers.PointMagneticFluxTimeDerivative(rx_locations[0], times, orientation="z")
    src = tdem.sources.LineCurrent([rx], location=source_locations, waveform=tdem.sources.StepOffWaveform())
    sim = tdem.Simulation1DLayered(survey=tdem.Survey([src]), sigmaMap=maps.IdentityMap(nP=1),
                                   n_points_per_path=n_points_per_path)
    return np.abs(sim.dpred(np.r_[sigma_hs]))
d_1d_square = run_1d_square()      # the reference everything is scored against

# background tensor mesh for mapping conductivity onto the octree by volume averaging
# (mirrors forward_farquarson.py; the validated approach). A naive sharp z<0 step is ~2x worse.
_hx = np.ones(40)*5; _hy = np.ones(30)*5; _hz = np.ones(31)*5
_mt = TensorMesh([_hx, _hy, _hz], x0=[-_hx.sum()/2, -_hy.sum()/2, -_hz.sum()+5])
_sig_bg = np.ones(_mt.n_cells)*sigma_hs; _sig_bg[_mt.cell_centers[:, 2] >= 0] = 1e-8


# ---- mesh + time-steps ------------------------------------------------------------------------
def build_octree(core=(10., 10., 5.), padding=5000., depth_core=100., refine="full", buf=200.):
    locs = np.vstack((source_locations, rx_locations))
    x = np.linspace(locs[:, 0].min()-buf, locs[:, 0].max()+buf)
    y = np.linspace(locs[:, 1].min()-buf, locs[:, 1].max()+buf)
    dem = utils.ndgrid(x, y, np.r_[0.0])
    mesh = dis_utils.mesh_builder_xyz(dem, list(core), padding_distance=[padding]*3,
                                      base_mesh=None, depth_core=depth_core, mesh_type='tree')
    if refine == "full":
        mesh.refine_surface(dem, -1, padding_cells_by_level=[[0, 0, 1], [0, 0, 15]], finalize=False)
        mesh.refine_line(source_locations, -1, finalize=False)
        mesh.refine_points(rx_locations, -1, finalize=False)
    elif refine == "light":
        mesh.refine_surface(dem, -1, padding_cells_by_level=[[0, 0, 1], [0, 0, 4]], finalize=False)
        mesh.refine_line(source_locations, -1, finalize=False)
        mesh.refine_points(rx_locations, -1, finalize=False)
    elif refine == "src_only":
        mesh.refine_line(source_locations, -1, finalize=False)
        mesh.refine_points(rx_locations, -1, finalize=False)
    else:
        raise ValueError(refine)
    mesh.finalize()
    return mesh


def gen_steps(rate, start, n_per_step, t_end=0.013):
    """Geometric ramp: n_per_step steps at each size, size growing by `rate`, until cumulative >= t_end.
    Distinct dt (= factorizations) equals the number of levels."""
    steps = []; dt = start; t = 0.0
    while t < t_end:
        for _ in range(n_per_step):
            steps.append(dt); t += dt
        dt *= rate
    return np.array(steps)


# ---- cached, timed 3D square-loop run ---------------------------------------------------------
CACHE = "explore_cache.npz"
_cache = {k: np.load(CACHE, allow_pickle=True)[k] for k in np.load(CACHE, allow_pickle=True).files} \
    if os.path.exists(CACHE) else {}


def _key(cfg):
    h = hashlib.md5(repr(sorted(cfg.items())).encode())
    return "k" + h.hexdigest()[:16]


def run_square(cfg, verbose=True):
    """cfg keys: core, padding, depth_core, refine, rate, start, n_per_step, t_end.
    Returns dict: d, dpred_s, n_cells, n_dt, n_steps, max_err, mean_err."""
    key = _key(cfg)
    if key in _cache:
        d, meta = _cache[key], _cache[key + "_m"]
        n_cells, n_dt, n_steps, dpred_s = int(meta[0]), int(meta[1]), int(meta[2]), float(meta[3])
        cached = True
    else:
        mesh = build_octree(cfg["core"], cfg["padding"], cfg["depth_core"], cfg["refine"],
                            buf=cfg.get("buf", 200))
        if cfg.get("sigma", "volavg") == "volavg":
            sig = volume_average(_mt, mesh) @ _sig_bg
        else:
            sig = np.ones(mesh.n_cells)*1e-8; sig[mesh.cell_centers[:, 2] < 0.0] = sigma_hs
        ts = gen_steps(cfg["rate"], cfg["start"], cfg["n_per_step"], cfg["t_end"])
        rx = time_domain.Rx.PointMagneticFluxTimeDerivative(rx_locations, times, "z")
        src = time_domain.Src.LineCurrent([rx], location=source_locations)
        sim = time_domain.Simulation3DElectricField(
            mesh, sigmaMap=maps.IdentityMap(nP=mesh.n_cells),
            survey=time_domain.Survey([src]), time_steps=ts, solver=Solver)
        t0 = _time.time(); d = np.abs(sim.dpred(sig)); dpred_s = _time.time() - t0
        n_cells, n_dt, n_steps = mesh.n_cells, len(np.unique(ts)), len(ts)
        _cache[key] = d
        _cache[key + "_m"] = np.array([n_cells, n_dt, n_steps, dpred_s], dtype=float)
        np.savez(CACHE, **_cache); cached = False
    r = d/d_1d_square
    out = dict(d=d, dpred_s=dpred_s, n_cells=n_cells, n_dt=n_dt, n_steps=n_steps,
               max_err=np.abs(r-1).max()*100, mean_err=(r.mean()-1)*100, cfg=cfg)
    if verbose:
        tag = "cached" if cached else f"{dpred_s:5.1f}s"
        print(f"  {n_cells/1e3:5.1f}k cells | {n_dt:2d} dt, {n_steps:3d} steps | {tag:>7s} | "
              f"max|err| {out['max_err']:4.1f}%  mean {out['mean_err']:+4.1f}%  "
              f"[core{tuple(cfg['core'])} pad{cfg['padding']:.0f} dc{cfg['depth_core']:.0f} "
              f"{cfg['refine']} x{cfg['rate']}/{cfg['start']:.0e}/{cfg['n_per_step']}]")
    return out


def cfg(core=(10., 10., 5.), padding=5000., depth_core=100., refine="full",
        rate=1.4, start=3e-7, n_per_step=6, t_end=0.013, sigma="volavg"):
    return dict(core=tuple(float(c) for c in core), padding=float(padding),
                depth_core=float(depth_core), refine=refine, rate=float(rate),
                start=float(start), n_per_step=int(n_per_step), t_end=float(t_end), sigma=sigma)
