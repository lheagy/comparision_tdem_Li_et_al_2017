import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Half-space TDEM: the time-stepping fix, validated against Ward & Hohmann

Reproduces **every figure and number** in `report.pdf`. The headline result is Figure 1: a one-line
time-stepping change moves the committed 3D SimPEG half-space response from +7–24 % high onto the
Ward & Hohmann analytic. The rest of the notebook follows one loop shape at a time to avoid circle/square
confusion:

1. **The Ward & Hohmann analytic** half-space response; SimPEG's 1D *circular* loop reproduces it to ~1e-3 %.
2. **The committed solution** (`half_tem_faquarson.csv`) agrees with Ward & Hohmann
   to ~2 % (this is the corrected file; the original is `…orig_backup`).
3. **Circular loop — the fix, isolated.** A 3D octree *circular*-loop convergence study shows the
   time-step ramp, not the mesh, is the dominant error; the residual few-% is a source-dependent octree
   spatial floor — verified *not* a bug in the circular-loop code (the initial field converges to the
   closed form with mesh refinement).
4. **Square loop — the actual benchmark.** The committed CSV is a *square* `LineCurrent`; the residual
   early-time ~1.8 % vs the circular analytic is real square-vs-circle geometry, not error.
5. **Reaching <1 %, the spatial floor, and a note on mesh design.**

**Naming.** "SimPEG" labels only curves we *compute by running SimPEG*. The **committed solution** is
loaded from `half_tem_faquarson.csv` (itself produced by a SimPEG 3D run, `forward_farquarson.py`; our
square re-runs reproduce it to 0.08 %).

**Caching.** Every 3D solve is cached to `tdem_results_cache.npz` keyed by (loop, mesh, time-steps), so the
first run takes ~11 min and every later rebuild loads from disk. Delete that file to force a recompute
(do so if you change the mesh builder or physics, which are not part of the key).
""")

# ---------------------------------------------------------------- 0. Setup
md("## 0. Setup")

code(r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings, time as _time, os, hashlib; warnings.filterwarnings("ignore")
from scipy.special import erf
from scipy.constants import mu_0

from discretize import TensorMesh
from discretize import utils as dis_utils
from discretize.utils import volume_average
from simpeg import utils, maps
from simpeg.electromagnetics import time_domain
import simpeg.electromagnetics.time_domain as tdem


def get_solver():
    '''Pick whatever fast direct solver is installed (Pardiso > Mumps > LU).'''
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
print("Using direct solver:", Solver.__name__)""")

code(r"""# Exact software environment this notebook was run in
import simpeg
simpeg.Report()""")

code(r"""# survey geometry: 100 x 100 m square loop, central dBz/dt receiver, step-off
dx = dy = 100.0
z_surface = 0.0
source_locations = np.array([[-dx/2, -dy/2, 0.], [-dx/2,  dy/2, 0.],
                             [ dx/2,  dy/2, 0.], [ dx/2, -dy/2, 0.], [-dx/2, -dy/2, 0.]])
rx_locations = np.array([[0., 0., 0.]])
times   = np.logspace(-4, -2, 21)
sigma_hs = 0.1
a = np.sqrt(dx*dy/np.pi)     # equal-area circular-loop radius
print(f"equal-area radius a = {a:.3f} m")""")

code(r"""# Ward & Hohmann analytic reference, and the SimPEG 1D layered reference solutions
def analytic_halfspace(times, sigma, radius):
    '''|dBz/dt| at loop centre on a homogeneous half-space, step-off (Ward & Hohmann 1988).'''
    theta = np.sqrt(mu_0 * sigma / (4.0 * times)); ta = theta * radius
    return np.abs(1.0/(sigma*radius**3) * (3*erf(ta) - 2/np.sqrt(np.pi)*ta*(3+2*ta**2)*np.exp(-ta**2)))


def run_1d(loop="square", n_points_per_path=10, radius=None):
    rx = tdem.receivers.PointMagneticFluxTimeDerivative(rx_locations[0], times, orientation="z")
    if loop == "square":
        src = tdem.sources.LineCurrent([rx], location=source_locations,
                                       waveform=tdem.sources.StepOffWaveform())
        sim = tdem.Simulation1DLayered(survey=tdem.Survey([src]), sigmaMap=maps.IdentityMap(nP=1),
                                       n_points_per_path=n_points_per_path)
    else:
        src = tdem.sources.CircularLoop([rx], location=np.r_[0., 0., 0.], radius=radius or a,
                                        waveform=tdem.sources.StepOffWaveform())
        sim = tdem.Simulation1DLayered(survey=tdem.Survey([src]), sigmaMap=maps.IdentityMap(nP=1))
    return np.abs(sim.dpred(np.r_[sigma_hs]))


d_anal      = analytic_halfspace(times, sigma_hs, a)
d_1d_square = run_1d("square")
d_1d_circle = run_1d("circle")
print("analytic + 1D layered references computed.")""")

# ---------------------------------------------------------------- Hero figure
md(r"""## Figure 1 (headline): a one-line time-stepping fix

The committed 3D SimPEG half-space solution before (**original**) and after (**improved**) the
time-stepping fix, against the **Ward & Hohmann analytic**. Both curves are loaded directly from the
committed CSVs (`…orig_backup` and `half_tem_faquarson.csv`).
""")

code(r"""# FIGURE ONLY — loads the two committed CSVs (no solve)
_orig = pd.read_csv("half_tem_faquarson.csv.orig_backup")
_impr = pd.read_csv("half_tem_faquarson.csv")
_t = _impr["times_s"].values
_an = analytic_halfspace(_t, sigma_hs, a)
_do = np.abs(_orig["voltage_V_Am2"].values)
_di = np.abs(_impr["voltage_V_Am2"].values)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
ax[0].loglog(_t, _an, 'k-',  lw=3.0, label='analytic', zorder=1)
ax[0].loglog(_t, _do, 'C3s--', ms=6, lw=1.6, label='original', zorder=2)
ax[0].loglog(_t, _di, 'C0o-',  ms=5, lw=1.6, label='improved 3D SimPEG', zorder=3)
ax[0].set_xlabel('time (s)'); ax[0].set_ylabel(r'$|\partial B_z/\partial t|$  (V/A·m$^2$)')
ax[0].grid(True, which='both', alpha=.4); ax[0].legend(fontsize=10)
ax[0].set_title('Half-space TDEM step-off response')
ax[1].semilogx(_t, _do/_an, 'C3s--', ms=6, lw=1.6,
               label=f'original (mean {(_do/_an).mean():.2f}, +{(_do/_an-1).max()*100:.0f}% max)')
ax[1].semilogx(_t, _di/_an, 'C0o-', ms=5, lw=1.6,
               label=f'improved (mean {(_di/_an).mean():.3f}, ±{np.abs(_di/_an-1).max()*100:.0f}%)')
ax[1].axhspan(0.98, 1.02, color='green', alpha=.12, label='±2% band')
ax[1].axhline(1, color='k', lw=1.5); ax[1].set_ylim(0.86, 1.3)

# rug of actual backward-Euler step times (cumulative); the early-time zig-zag tracks where the coarse
# original steps fall relative to the observation channels. BOLD red lines mark where the original step
# WIDTH changes (the ×2 level boundaries).
def _step_times(n, r, s, p):
    sizes = np.hstack([np.ones(p)*s*r**i for i in range(n)])
    tcum = np.cumsum(sizes)
    return tcum, tcum[np.where(np.diff(sizes) > 0)[0]]
_to, _wc = _step_times(11, 2.0, 1e-6, 5)
_ti, _   = _step_times(26, 1.4, 3e-7, 6)
_win = lambda x: x[(x >= _t.min()) & (x <= _t.max())]
ax[1].plot(_win(_to), np.full_like(_win(_to), 0.895), '|', color='C3', ms=10, mew=1.0,
           alpha=.45, label=f'original step times ({len(_to)})')
ax[1].vlines(_win(_wc), 0.872, 0.918, color='C3', lw=2.6, alpha=.95, zorder=5,
             label='original step-width change (×2)')
ax[1].plot(_win(_ti), np.full_like(_win(_ti), 0.94), '|', color='C0', ms=10, mew=1.0,
           alpha=.55, label=f'improved step times ({len(_ti)})')
ax[1].set_xlabel('time (s)'); ax[1].set_ylabel('ratio to analytic')
ax[1].grid(True, which='both', alpha=.4); ax[1].legend(fontsize=8.2, loc='upper left')
ax[1].set_title('A one-line time-stepping fix: +7–24% → within ~2%')
plt.tight_layout(); plt.savefig('fig_hero.png', dpi=140); plt.show()""")

# ---------------------------------------------------------------- 1. Ward & Hohmann
md(r"""## 1. The Ward & Hohmann analytic half-space response

The vertical $\partial B_z/\partial t$ at the centre of a loop on a homogeneous half-space after a current
step-off (Ward & Hohmann, 1988) is

$$\frac{\partial b_z}{\partial t} = -\frac{1}{\sigma a^3}\left[3\,\mathrm{erf}(\theta a)
  - \frac{2}{\sqrt{\pi}}\,\theta a\,(3+2\theta^2 a^2)\,e^{-\theta^2 a^2}\right],
  \qquad \theta = \sqrt{\frac{\mu_0\sigma}{4t}},$$

with $\sigma=0.1$ S/m and equal-area radius $a=\sqrt{A/\pi}=56.42$ m for the $100\times100$ m loop. This is
the closed form for a **circular** loop. `Simulation1DLayered` with a *circular* loop of radius $a$
reproduces it to ~1e-3 %, so SimPEG's 1D code is the trustworthy reference; the actual *square*
`LineCurrent` differs by ~1.8 % early (geometry — Section 4).
""")

code(r"""# COMPUTE / report the 1D validation numbers (references computed in Setup)
print(f"1D circular loop vs analytic : max error = {np.abs(d_1d_circle/d_anal-1).max()*100:.4f} %")
print(f"1D square   loop vs analytic : max error = {np.abs(d_1d_square/d_anal-1).max()*100:.2f} % "
      f"(square-vs-circle geometry; see Section 4)")""")

md("### Figure 2 — Ward & Hohmann analytic, and the SimPEG 1D circular-loop reproduction")

code(r"""# FIGURE ONLY
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].loglog(times, d_anal, 'k-', lw=2.5, label='Ward & Hohmann analytic (circle, r=a)')
ax[0].loglog(times, d_1d_circle, 'C0o', ms=5, label='SimPEG 1D layered (circular loop)')
ax[0].set_xlabel('time (s)'); ax[0].set_ylabel(r'$|\partial B_z/\partial t|$ (V/A·m$^2$)')
ax[0].grid(True, which='both', alpha=.4); ax[0].legend(fontsize=9)
ax[0].set_title('Ward & Hohmann half-space response')
ax[1].semilogx(times, (d_1d_circle/d_anal-1)*1e2, 'C0o-', ms=5,
               label=f'max |error| = {np.abs(d_1d_circle/d_anal-1).max()*1e2:.1e} %')
ax[1].axhline(0, color='k', lw=1.5); ax[1].set_xlabel('time (s)')
ax[1].set_ylabel('1D circle / analytic − 1  [%]')
ax[1].grid(True, which='both', alpha=.4); ax[1].legend(fontsize=9)
ax[1].set_title('SimPEG 1D circular loop reproduces the closed form')
plt.tight_layout(); plt.savefig('fig_wardhohmann_1d.png', dpi=130); plt.show()""")

# ---------------------------------------------------------------- 2. Validation of the file
md(r"""## 2. The committed solution agrees with Ward & Hohmann

The solution of record — the data in `half_tem_faquarson.csv`, loaded straight from disk (no SimPEG run) —
matches the closed-form Ward & Hohmann response to within ~2 % across all 21 channels (mean ratio 0.996;
the **improved** curve in Figure 1). The pre-fix original (`…orig_backup`) sat at mean 1.187 (+7–24 %).
""")

code(r"""# LOAD: confirm the committed (corrected) file vs analytic; contrast with the original backup
csv = pd.read_csv("half_tem_faquarson.csv")
t_csv = csv["times_s"].values
d_farq = np.abs(csv["voltage_V_Am2"].values)
d_wh_csv = analytic_halfspace(t_csv, sigma_hs, a)
r_impr = d_farq / d_wh_csv
print(f"improved half_tem_faquarson.csv vs Ward & Hohmann: mean {r_impr.mean():.3f}, "
      f"range {r_impr.min():.3f}-{r_impr.max():.3f}  (within ~2%)")
if os.path.exists("half_tem_faquarson.csv.orig_backup"):
    d_orig = np.abs(pd.read_csv("half_tem_faquarson.csv.orig_backup")["voltage_V_Am2"].values)
    r_orig = d_orig / d_wh_csv
    print(f"original .orig_backup           vs Ward & Hohmann: mean {r_orig.mean():.3f}, "
          f"range {r_orig.min():.3f}-{r_orig.max():.3f}  (+7-24%)")""")

# ---------------------------------------------------------------- 3. Circle: the fix isolated
md(r"""## 3. Circular loop: the time-stepping fix, isolated

To isolate the *temporal* error from any geometry, we run the 3D octree with a **circular** loop
(radius $a$) and compare against the **exact** analytic. Refining the **time-steps** on the original mesh
drives the error down sharply; refining the **mesh** ($3.3\times$ cells) on the original time-steps barely
helps. The original $\times2$ ramp (55 steps) is too coarse for backward-Euler; the fix is a gentler
$\times1.4$ ramp.

The one-line change in `forward_farquarson.py`:

```
- generate_time_steps(n_constant_steps=11, increase_rate=2,   start_time_step=1e-6, n_per_step=5)  # 55 steps
+ generate_time_steps(n_constant_steps=26, increase_rate=1.4, start_time_step=3e-7, n_per_step=6)  # 156 steps
```

The circle does **not** reach the analytic exactly even after the fix — it bottoms out a few percent high.
That residual is a *source-dependent octree spatial floor*, not a time-stepping issue and not a bug in the
circular-loop code (verified below).
""")

code(r"""# --- 3D octree machinery (mirrors forward_farquarson.py), with on-disk caching ----------
hx = np.ones(40)*5; hy = np.ones(30)*5; hz = np.ones(31)*5
mesh_tensor = TensorMesh([hx, hy, hz], x0=[-hx.sum()/2, -hy.sum()/2, -hz.sum()+5])
active = mesh_tensor.cell_centers[:, 2] < 0.0
sigma_tensor = np.ones(mesh_tensor.n_cells)*sigma_hs; sigma_tensor[~active] = 1e-8


def build_octree(h):
    buf = 200
    locs = np.vstack((source_locations, rx_locations))
    x = np.linspace(locs[:, 0].min()-buf, locs[:, 0].max()+buf)
    y = np.linspace(locs[:, 1].min()-buf, locs[:, 1].max()+buf)
    dem = utils.ndgrid(x, y, np.r_[z_surface])
    mesh = dis_utils.mesh_builder_xyz(dem, h, padding_distance=[5000, 5000, 5000],
                                      base_mesh=None, depth_core=100, mesh_type='tree')
    mesh.refine_surface(dem, -1, padding_cells_by_level=[[0, 0, 1], [0, 0, 15]], finalize=False)
    mesh.refine_line(source_locations, -1, finalize=False)
    mesh.refine_points(rx_locations, -1, finalize=False)
    mesh.finalize()
    return mesh


def make_time_steps(n_constant_steps, increase_rate, start_time_step, n_per_step):
    return np.hstack([np.ones(n_per_step)*start_time_step*increase_rate**i
                      for i in range(n_constant_steps)])


def octree_source(loop, mesh):
    rx = time_domain.Rx.PointMagneticFluxTimeDerivative(rx_locations, times, "z")
    if loop == "circle":
        return rx, time_domain.Src.CircularLoop([rx], location=np.r_[0., 0., 0.], radius=a)
    return rx, time_domain.Src.LineCurrent([rx], location=source_locations)


# on-disk cache: every solve runs once, then loads from disk
CACHE_FILE = "tdem_results_cache.npz"
_cache = {k: np.load(CACHE_FILE)[k] for k in np.load(CACHE_FILE).files} if os.path.exists(CACHE_FILE) else {}

def _ckey(*parts):
    hsh = hashlib.md5()
    for p in parts:
        if isinstance(p, np.ndarray):
            hsh.update(np.ascontiguousarray(p, dtype=float).tobytes())
        else:
            hsh.update(repr(p).encode())
    return "k" + hsh.hexdigest()[:14]

def _cache_put(key, d, ncells):
    _cache[key] = d; _cache[key + "_nc"] = np.array(ncells); np.savez(CACHE_FILE, **_cache)


def run_3d(name, h, ts, loop="square", store=None):
    '''3D octree dB/dt (cached); loop='circle' uses a CircularLoop, 'square' a LineCurrent.'''
    key = _ckey("octree", loop, tuple(h), np.asarray(ts))
    cached = key in _cache
    if cached:
        d, nc = _cache[key], int(_cache[key + "_nc"])
    else:
        t0 = _time.time(); mesh = build_octree(h)
        sig = volume_average(mesh_tensor, mesh) @ sigma_tensor
        rx, src = octree_source(loop, mesh)
        sim = time_domain.Simulation3DElectricField(
            mesh, sigmaMap=maps.IdentityMap(nP=mesh.n_cells),
            survey=time_domain.Survey([src]), time_steps=ts, solver=Solver)
        d = np.abs(sim.dpred(sig)); nc = mesh.n_cells
        _cache_put(key, d, nc); dt = _time.time() - t0
    if store is not None:
        store[name] = d
    NCELLS[(loop, name)] = nc
    ref = d_anal if loop == "circle" else d_1d_square
    print(f"  [{loop:6s}|{name:10s}] {nc/1e3:4.0f}k cells, {len(ts):3d} steps, "
          f"{'cached' if cached else f'{dt:3.0f}s':>6s} | mean/ref = {(d/ref).mean():.3f}")
    return d

TS = {
    "baseline":  make_time_steps(11, 2.00, 1e-6, 5),   # original, x2,    55 steps
    "time_fine": make_time_steps(18, 1.50, 1e-6, 5),   #           x1.5,  90 steps
    "fix":       make_time_steps(26, 1.40, 3e-7, 6),   # the fix,  x1.4, 156 steps
    "r130":      make_time_steps(33, 1.30, 1e-7, 6),   #           x1.3, 198 steps
    "r125":      make_time_steps(40, 1.25, 7e-8, 8),   #           x1.25,320 steps
}
RC = {}   # circular-loop 3D results
RS = {}   # square-loop  3D results
NCELLS = {}
print("machinery ready.")""")

code(r"""# COMPUTE: circular-loop convergence study vs EXACT analytic (cached after first run)
print("3D octree circular-loop solves (ratio = mean to exact analytic):")
run_3d("baseline",  [10, 10, 5],  TS["baseline"],  "circle", RC)
run_3d("time_fine", [10, 10, 5],  TS["time_fine"], "circle", RC)
run_3d("fix",       [10, 10, 5],  TS["fix"],       "circle", RC)
run_3d("mesh_fine", [5, 5, 2.5],  TS["baseline"],  "circle", RC)
run_3d("mesh_time", [5, 5, 2.5],  TS["time_fine"], "circle", RC)

print("\nmean ratio to analytic:")
for k in ["baseline", "time_fine", "fix", "mesh_fine", "mesh_time"]:
    r = RC[k]/d_anal
    print(f"  {k:10s}: mean {r.mean():.3f}  range {r.min():.3f}-{r.max():.3f}")""")

md("### Figure 3 — convergence study (circular loop vs exact analytic)")

code(r"""# FIGURE ONLY
order = [("baseline","coarse mesh + orig ×2 steps (55)","o-"),
         ("time_fine","coarse mesh + ×1.5 steps (90)","s-"),
         ("fix","coarse mesh + ×1.4 steps (156, the fix)","^-"),
         ("mesh_fine","FINE mesh + orig ×2 steps","x--"),
         ("mesh_time","fine mesh + ×1.5 steps","d-")]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].loglog(times, d_anal, 'k-', lw=2.5, label='closed-form analytic')
for k, lab, st in order:
    ax[0].loglog(times, RC[k], st, ms=4, label=lab)
ax[0].set_xlabel('time (s)'); ax[0].set_ylabel(r'$|\partial B_z/\partial t|$')
ax[0].grid(True, which='both', alpha=.4); ax[0].legend(fontsize=7.5)
ax[0].set_title('SimPEG 3D octree (circular loop) vs analytic')
for k, lab, st in order:
    ax[1].semilogx(times, RC[k]/d_anal, st, ms=4, label=lab)
ax[1].axhline(1, color='k', lw=2); ax[1].set_ylim(0.95, 1.35)
ax[1].set_xlabel('time (s)'); ax[1].set_ylabel('ratio to analytic'); ax[1].grid(True, which='both', alpha=.4)
ax[1].legend(fontsize=7.5); ax[1].set_title('Time-steps close most of the gap; a spatial floor remains')
plt.tight_layout(); plt.savefig('fig_convergence_timesteps.png', dpi=130); plt.show()""")

md("### Figure 4 — circular loop, before/after the fix")

code(r"""# FIGURE ONLY
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
ax[0].loglog(times, d_anal, 'k-', lw=2.5, label='analytic half-space (Ward & Hohmann)')
ax[0].loglog(times, d_1d_circle, 'C2o', ms=4, label='SimPEG 1D layered (circle)')
ax[0].loglog(times, RC["baseline"], 'C3s-', ms=4, label='SimPEG 3D octree — ORIGINAL steps')
ax[0].loglog(times, RC["fix"], 'C0^-', ms=4, label='SimPEG 3D octree — FIXED steps')
ax[0].set_xlabel('time (s)'); ax[0].set_ylabel(r'$|\partial B_z/\partial t|$ (V/A·m$^2$)')
ax[0].grid(True, which='both', alpha=.4); ax[0].legend(fontsize=8); ax[0].set_title('Half-space TEM response (circle)')
ax[1].semilogx(times, d_1d_circle/d_anal, 'C2o', ms=4, label='1D layered (~1e-3%)')
ax[1].semilogx(times, RC["baseline"]/d_anal, 'C3s-', ms=4,
               label=f'3D ORIGINAL (mean {(RC["baseline"]/d_anal).mean():.2f})')
ax[1].semilogx(times, RC["fix"]/d_anal, 'C0^-', ms=4,
               label=f'3D FIXED (mean {(RC["fix"]/d_anal).mean():.3f})')
ax[1].axhline(1, color='k', lw=2); ax[1].set_ylim(0.95, 1.35)
ax[1].set_xlabel('time (s)'); ax[1].set_ylabel('ratio to analytic')
ax[1].grid(True, which='both', alpha=.4); ax[1].legend(fontsize=8)
ax[1].set_title('Fix removes the temporal bias; octree spatial floor remains')
plt.tight_layout(); plt.savefig('fig_before_after_fix.png', dpi=130); plt.show()""")

md(r"""### Is the circular-loop source buggy? No — the initial field converges to the closed form

For a step-off the initial condition is the magnetostatic loop field; at the centre of a circular loop
$B_z = \mu_0 I / (2a)$ **exactly**. We compute the *discretized* initial $B_z$ at the receiver (no
time-stepping) on a sequence of refined meshes. It converges monotonically to the closed form — textbook
spatial discretization, not a formulation error. (`CircularLoop` uses the finite-loop
`CircularLoopWholeSpace` vector potential, and the 1D circular loop is exact to 1e-3 %.)
""")

code(r"""# COMPUTE: discretized initial Bz at loop centre vs closed form, across mesh refinement (cached)
Bz_closed = mu_0*1.0/(2*a)

def initial_Bz_circle(h):
    key = _ckey("initbz", tuple(h))
    if key in _cache:
        return int(_cache[key + "_nc"]), float(_cache[key])
    mesh = build_octree(h)
    rx, src = octree_source("circle", mesh)
    sim = time_domain.Simulation3DElectricField(
        mesh, sigmaMap=maps.IdentityMap(nP=mesh.n_cells), survey=time_domain.Survey([src]))
    b0 = np.asarray(src.bInitial(sim)).reshape(-1)
    P = mesh.get_interpolation_matrix(rx_locations, 'faces_z')
    bz = b0[mesh.n_faces_x+mesh.n_faces_y:] if P.shape[1] == mesh.n_faces_z else b0
    val = float((P @ bz)[0]); _cache_put(key, np.array(val), mesh.n_cells)
    return mesh.n_cells, val

init_cells, init_ratio = [], []
print(f"closed-form initial Bz at centre = {Bz_closed:.4e} T")
for h in [[15,15,7.5],[10,10,5],[7,7,3.5],[5,5,2.5],[3,3,1.5]]:
    nc, bz = initial_Bz_circle(h)
    init_cells.append(nc/1e3); init_ratio.append(bz/Bz_closed)
    print(f"  h={str(h):14s} {nc/1e3:5.0f}k cells | ratio to closed form = {bz/Bz_closed:.4f} "
          f"({(bz/Bz_closed-1)*100:+.2f}%)")""")

md("### Figure 5 — the circular-loop initial field converges with mesh refinement (not a bug)")

code(r"""# FIGURE ONLY
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.semilogx(init_cells, (np.array(init_ratio)-1)*100, 'C0o-', ms=7)
ax.axhline(0, color='k', lw=1.5, label='exact $\\mu_0 I/(2a)$')
for xc, yr in zip(init_cells, init_ratio):
    ax.annotate(f'{(yr-1)*100:+.2f}%', (xc, (yr-1)*100), textcoords='offset points',
                xytext=(0, 8), ha='center', fontsize=8)
ax.set_xlabel('mesh size (1000s of cells)'); ax.set_ylabel('initial $B_z$ at centre / closed form − 1  [%]')
ax.grid(True, which='both', alpha=.4); ax.legend(fontsize=9)
ax.set_title('CircularLoop initial field → closed form as the octree refines\n(spatial discretization, not a code bug)')
plt.tight_layout(); plt.savefig('fig_initial_field_convergence.png', dpi=130); plt.show()""")

# ---------------------------------------------------------------- 4. Square: the benchmark
md(r"""## 4. Square loop: the actual benchmark, and why the residual is geometry

The committed `half_tem_faquarson.csv` is a **square** `LineCurrent` loop. We re-run that geometry on the
same octree, before and after the fix, and confirm the re-runs reproduce the committed CSVs. The remaining
early-time ~1.8 % when scored against the *circular* analytic is real square-vs-circle geometry, not error.
""")

code(r"""# COMPUTE: square-loop original vs fixed time-stepping; confirm they reproduce the committed CSVs
print("3D octree square-loop solves (ratio = mean to 1D square reference):")
run_3d("baseline", [10, 10, 5], TS["baseline"], "square", RS)
run_3d("fix",      [10, 10, 5], TS["fix"],      "square", RS)

for fname, key in [("half_tem_faquarson.csv.orig_backup", "baseline"),
                   ("half_tem_faquarson.csv", "fix")]:
    if os.path.exists(fname):
        c = np.abs(pd.read_csv(fname)["voltage_V_Am2"].values)
        print(f"{fname:36s}: max diff vs RS['{key}'] = {np.abs(RS[key]/c-1).max()*100:.2f} %")""")

md("### Figure 6 — localizing the discrepancy: 1D square correct, original 3D square off")

code(r"""# FIGURE ONLY
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].loglog(times, RS["baseline"], 'C0-o', ms=3, label='SimPEG 3D octree (square, original)')
ax[0].loglog(times, d_1d_square, 'k--', label='SimPEG 1D layered (square)')
ax[0].loglog(times, d_anal, 'r:', lw=2, label='analytic (equal-area circle)')
ax[0].set_xlabel('time (s)'); ax[0].set_ylabel(r'$|\partial B_z/\partial t|$ (V/A·m$^2$)')
ax[0].grid(True, which='both', alpha=.4); ax[0].legend(fontsize=8); ax[0].set_title('Half-space TEM response')
ax[1].semilogx(times, RS["baseline"]/d_anal, 'C0-o', ms=3, label='3D octree (square) / analytic')
ax[1].semilogx(times, d_1d_square/d_anal, 'k--', label='1D layered (square) / analytic')
ax[1].axhline(1, color='r', ls=':'); ax[1].set_ylim(0.9, 1.28)
ax[1].set_xlabel('time (s)'); ax[1].set_ylabel('ratio to analytic')
ax[1].grid(True, which='both', alpha=.4); ax[1].legend(fontsize=8); ax[1].set_title('Relative error')
plt.tight_layout(); plt.savefig('fig_diagnosis_1d_vs_3d.png', dpi=130); plt.show()""")

md(r"""**Why the 1.8 % is geometry, not error.** The wire integration is converged (refining
`n_points_per_path` changes the result by <0.07 %), and a circle reproduces the analytic to 0.0007 %.
There is no elementary closed form for a square loop — the "analytic" curve is a *circle*. A square matches
its **equal-area** circle only in the late-time point-dipole limit; at early time it depends on the actual
distance to the wire (sides at 50 m, corners at 70.7 m, versus a circle at 56.4 m everywhere).
""")

code(r"""# COMPUTE: wire-integration convergence + square vs three reference circles (1D, cheap)
sq_ref = run_1d("square", n_points_per_path=200)
for npp in (3, 10, 40, 100):
    d = run_1d("square", n_points_per_path=npp)
    print(f"  n_points_per_path={npp:3d}: max diff vs converged (200 pts) = {np.abs(d/sq_ref-1).max()*100:.4f} %")

S    = run_1d("square", n_points_per_path=20)
c_in = run_1d("circle", radius=50.0)
c_eq = run_1d("circle", radius=a)
c_out= run_1d("circle", radius=70.71)
print("geometry arrays computed.")""")

md("### Figure 7 — square vs inscribed / equal-area / circumscribed circles")

code(r"""# FIGURE ONLY
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].plot(source_locations[:, 0], source_locations[:, 1], 'k-', lw=2, label='square loop (100×100 m)')
th = np.linspace(0, 2*np.pi, 200)
for rad, c, ls, tag in [(50, 'C0', ':', 'inscribed'), (a, 'C2', '-', 'equal area'), (70.71, 'C3', '--', 'circumscribed')]:
    ax[0].plot(rad*np.cos(th), rad*np.sin(th), c, ls=ls, lw=1.6, label=f'circle r={rad:.1f} m ({tag})')
ax[0].plot(0, 0, 'kx', ms=8); ax[0].set_aspect(1); ax[0].set_xlim(-80, 80); ax[0].set_ylim(-80, 95)
ax[0].set_xlabel('x (m)'); ax[0].set_ylabel('y (m)'); ax[0].legend(fontsize=8, loc='upper right')
ax[0].grid(alpha=.3); ax[0].set_title('Source geometry')
ax[1].semilogx(times, S/c_in, 'C0:', lw=1.8, label='vs inscribed (r=50)')
ax[1].semilogx(times, S/c_eq, 'C2-', lw=2.2, label='vs equal-area (r=56.4)')
ax[1].semilogx(times, S/c_out, 'C3--', lw=1.8, label='vs circumscribed (r=70.7)')
ax[1].axhline(1, color='k', lw=1); ax[1].set_ylim(0.6, 1.3)
ax[1].set_xlabel('time (s)'); ax[1].set_ylabel('square / circle (1D layered)')
ax[1].grid(True, which='both', alpha=.3); ax[1].legend(fontsize=8)
ax[1].set_title('Square ⟶ equal-area circle only in the\nlate-time (point-dipole) limit')
plt.tight_layout(); plt.savefig('fig_loop_geometry.png', dpi=130); plt.show()""")

# ---------------------------------------------------------------- 5. Sub-1% + mesh note
md(r"""## 5. Reaching <1 %, the spatial floor, and a note on mesh design

Scored against the correct same-geometry reference (1D layered **square** loop): the fix sits at mean
+0.2 % (max ~3 %), but pushing the steps finer overshoots — exposing the spatial floor a few % low. The
fix's great mean is partly a cancellation of a temporal high-bias and a spatial low-bias.
""")

code(r"""# COMPUTE: push square time-stepping past the fix (cached)
run_3d("r130", [10, 10, 5], TS["r130"], "square", RS)
run_3d("r125", [10, 10, 5], TS["r125"], "square", RS)

print("\nvs 1D layered SQUARE reference:")
for k, lab in [("fix","×1.4, 156 (fix)"), ("r130","×1.3, 198"), ("r125","×1.25, 320")]:
    e = RS[k]/d_1d_square - 1
    print(f"  {lab:16s}: mean {e.mean()*100:+.1f}%  max |err| {np.abs(e).max()*100:.1f}%")""")

md("### Figure 8 — refining time-steps shifts every channel downward")

code(r"""# FIGURE ONLY
series = [('baseline (×2, 55)', RS["baseline"], 'C3'),
          ('×1.4, 156  (fix)', RS["fix"], 'C0'),
          ('×1.3, 198', RS["r130"], 'C4'),
          ('×1.25, 320', RS["r125"], 'C5')]
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for lab, d, c in series:
    ax.semilogx(times, (d/d_1d_square-1)*100, 'o-', ms=3, color=c, label=lab)
ax.axhspan(-1, 1, color='green', alpha=.12, label='±1% band'); ax.axhline(0, color='k', lw=1)
ax.set_xlabel('time (s)'); ax.set_ylabel('error vs 1D layered (square) [%]'); ax.set_ylim(-12, 22)
ax.legend(fontsize=8, ncol=2); ax.grid(True, which='both', alpha=.3)
ax.set_title('Refining time-steps shifts every channel downward\n(temporal high-bias gives way to a spatial floor)')
plt.tight_layout(); plt.savefig('fig_timestep_refinement.png', dpi=130); plt.show()""")

md(r"""### Takeaway on time-stepping design

The residual floor is spatial (set by the mesh), and closing it to sub-1 % at every channel is a
mesh-refinement exercise, not a time-stepping one. The time-stepping lesson is the opposite of "use the
finest steps possible": the backward-Euler ramp must be **conservative** — gently growing, with enough
steps to resolve the early-time decay — but pushing it arbitrarily fine just trades a temporal high-bias
for the spatial floor. The $\times1.4$ / 156-step ramp is the balanced choice.
""")

nb["cells"] = cells
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
with open("halfspace_validation_and_timestep_fix.ipynb", "w") as f:
    nbf.write(nb, f)
print(f"wrote halfspace_validation_and_timestep_fix.ipynb with {len(cells)} cells")
