"""
Kalman Filter Missile Intercept Simulation
- Runs N_MISSIONS independent Monte Carlo missions with random target trajectories.
- Applies a discrete Kalman filter to noisy radar returns, computes intercept
  geometry, and reports averaged performance across all missions.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ── Config ────────────────────────────────────────────────────────────────────
# No random seed — every run produces different trajectories and statistics.
DT          = 0.5          # time step (seconds)
T_TOTAL     = 60.0         # mission duration (seconds)
SIGMA_R     = 50.0         # radar noise std dev (meters)
MISSILE_SPD = 800.0        # missile speed (m/s)
TARGET_SPD  = 300.0        # target nominal speed (m/s)
EVASION_Q   = 5.0          # random acceleration std dev (m/s²)
KILL_RADIUS = 300.0        # lethal radius for intercept success (m)
LAUNCH_T    = 10.0         # missile launch time (s)
LAUNCH_POS  = np.array([2000.0, -1000.0])
N_MISSIONS  = 15           # Monte Carlo runs

os.makedirs("results", exist_ok=True)
STEPS = int(T_TOTAL / DT)
TIME  = np.linspace(0, T_TOTAL, STEPS)

# ── 1. Target Simulation ─────────────────────────────────────────────────────
def simulate_target():
    """CV target with random initial heading and Gaussian evasion maneuvers."""
    states = np.zeros((STEPS, 4))  # [px, py, vx, vy]
    angle = np.random.uniform(0, 2 * np.pi)   # randomised each mission
    states[0] = [0, 0, TARGET_SPD * np.cos(angle), TARGET_SPD * np.sin(angle)]
    for k in range(1, STEPS):
        ax, ay = np.random.normal(0, EVASION_Q, 2)
        states[k, 0] = states[k-1, 0] + states[k-1, 2]*DT + 0.5*ax*DT**2
        states[k, 1] = states[k-1, 1] + states[k-1, 3]*DT + 0.5*ay*DT**2
        states[k, 2] = states[k-1, 2] + ax*DT
        states[k, 3] = states[k-1, 3] + ay*DT
    return states

# ── 2. Radar Measurements ────────────────────────────────────────────────────
def get_measurements(true_states):
    noise = np.random.normal(0, SIGMA_R, (STEPS, 2))
    return true_states[:, :2] + noise

# ── 3. Kalman Filter ─────────────────────────────────────────────────────────
def kalman_filter(measurements):
    """
    Discrete Kalman filter: state x = [px, py, vx, vy], obs z = [px, py].
    Q uses the Singer discrete-time acceleration noise model.
    """
    F = np.array([[1, 0, DT, 0],
                  [0, 1, 0, DT],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]])

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

    Q = EVASION_Q**2 * np.array([
        [DT**4/4, 0,       DT**3/2, 0      ],
        [0,       DT**4/4, 0,       DT**3/2],
        [DT**3/2, 0,       DT**2,   0      ],
        [0,       DT**3/2, 0,       DT**2  ],
    ])

    R = SIGMA_R**2 * np.eye(2)

    estimates = np.zeros((STEPS, 4))
    P = np.eye(4) * 5000.0
    estimates[0] = [measurements[0, 0], measurements[0, 1], 0.0, 0.0]

    for k in range(1, STEPS):
        x_p = F @ estimates[k-1]
        P_p = F @ P @ F.T + Q
        S   = H @ P_p @ H.T + R
        K   = P_p @ H.T @ np.linalg.inv(S)
        estimates[k] = x_p + K @ (measurements[k] - H @ x_p)
        P = (np.eye(4) - K @ H) @ P_p

    return estimates

# ── 4. Intercept Calculator ───────────────────────────────────────────────────
def find_intercept(kf_state, launch_pos, missile_speed):
    """
    Scan future time steps; return first (intercept_point, time_to_intercept)
    where the missile (constant speed, straight line) can reach the predicted
    target position in the allotted time.
    """
    px, py, vx, vy = kf_state
    mx, my = launch_pos
    for t in np.arange(DT, T_TOTAL, DT):
        tx, ty = px + vx*t, py + vy*t
        dist   = np.hypot(tx - mx, ty - my)
        if dist <= missile_speed * t:
            return np.array([tx, ty]), t
    return None, None

# ── 5. Single Mission ─────────────────────────────────────────────────────────
def run_mission():
    true_states  = simulate_target()
    measurements = get_measurements(true_states)
    kf_states    = kalman_filter(measurements)

    launch_step = int(LAUNCH_T / DT)
    ipt, t_ipt  = find_intercept(kf_states[launch_step], LAUNCH_POS, MISSILE_SPD)

    raw_err = np.linalg.norm(measurements - true_states[:, :2], axis=1)
    kf_err  = np.linalg.norm(kf_states[:, :2] - true_states[:, :2], axis=1)
    err_pct = (1 - kf_err.mean() / raw_err.mean()) * 100

    miss_path = miss_dist = actual_pos = None
    hit = False
    if ipt is not None:
        n_ipt      = int(t_ipt / DT)
        frac       = np.linspace(0, 1, n_ipt + 1)[:, None]
        miss_path  = LAUNCH_POS + frac * (ipt - LAUNCH_POS)
        actual_idx = min(launch_step + n_ipt, STEPS - 1)
        actual_pos = true_states[actual_idx, :2]
        miss_dist  = np.linalg.norm(ipt - actual_pos)
        hit        = miss_dist < KILL_RADIUS

    return dict(
        true_states=true_states, measurements=measurements, kf_states=kf_states,
        ipt=ipt, t_ipt=t_ipt, miss_path=miss_path,
        miss_dist=miss_dist, actual_pos=actual_pos, hit=hit,
        raw_err=raw_err, kf_err=kf_err, err_pct=err_pct,
    )

# ── 6. Run All Missions ───────────────────────────────────────────────────────
print(f"Running {N_MISSIONS} missile intercept missions...")
results = [run_mission() for _ in range(N_MISSIONS)]

hits        = sum(r["hit"] for r in results)
success_pct = hits / N_MISSIONS * 100
avg_err_pct = np.mean([r["err_pct"] for r in results])
avg_raw_err = np.mean([r["raw_err"].mean() for r in results])
avg_kf_err  = np.mean([r["kf_err"].mean() for r in results])
valid_dists = [r["miss_dist"] for r in results if r["miss_dist"] is not None]
avg_miss    = np.mean(valid_dists) if valid_dists else float("nan")

# Representative mission: first hit, or first mission if none
rep = next((r for r in results if r["hit"]), results[0])
launch_step = int(LAUNCH_T / DT)

# ── 7. Plot ───────────────────────────────────────────────────────────────────
C = dict(true="#2ecc71", meas="#e74c3c", kf="#3498db", msle="#f39c12")
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle(
    f"Kalman Filter · Missile Intercept — {N_MISSIONS} Missions  "
    f"| Success: {success_pct:.0f}%  |  Avg Error Reduction: {avg_err_pct:.1f}%",
    fontsize=13, fontweight="bold",
)

# ── Plot 1: Trajectory (representative mission) ──────────────────────────────
ax = axes[0, 0]
ts, meas, kf = rep["true_states"], rep["measurements"], rep["kf_states"]
ax.scatter(meas[:, 0], meas[:, 1], s=4, color=C["meas"],
           alpha=0.35, label=f"Radar (σ={SIGMA_R:.0f} m)")
ax.plot(*ts[:, :2].T, color=C["true"], lw=2.0, label="True trajectory")
ax.plot(*kf[:, :2].T, color=C["kf"],  lw=2.0, ls="--", label="Kalman estimate")
ax.plot(*ts[0, :2],  "g^", ms=10, label="Start")
ax.plot(*ts[-1, :2], "rs", ms=10, label="End")
ax.set_title("Plot 1: Trajectory Tracking (representative mission)")
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── Plot 2: Tracking Error (representative mission) ──────────────────────────
ax = axes[0, 1]
raw_err, kf_err = rep["raw_err"], rep["kf_err"]
ax.fill_between(TIME, raw_err, alpha=0.25, color=C["meas"])
ax.fill_between(TIME, kf_err,  alpha=0.40, color=C["kf"])
ax.plot(TIME, raw_err, color=C["meas"], lw=1.2, label="Raw radar error")
ax.plot(TIME, kf_err,  color=C["kf"],  lw=1.5, label="Kalman error")
ax.axhline(raw_err.mean(), color=C["meas"], ls="--", lw=1.5,
           label=f"Avg raw {raw_err.mean():.1f} m")
ax.axhline(kf_err.mean(),  color=C["kf"],  ls="--", lw=1.5,
           label=f"Avg KF  {kf_err.mean():.1f} m")
ax.set_title(f"Plot 2: Tracking Error — {rep['err_pct']:.1f}% reduction (this mission)")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Position Error (m)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── Plot 3: Intercept Geometry (representative mission) ──────────────────────
ax = axes[1, 0]
ipt, miss_path = rep["ipt"], rep["miss_path"]
actual_pos, miss_dist, hit = rep["actual_pos"], rep["miss_dist"], rep["hit"]
ax.plot(*ts[:, :2].T, color=C["true"], lw=2, alpha=0.8, label="Target path")
if ipt is not None:
    ax.plot(*miss_path.T, color=C["msle"], lw=2.5, label="Missile path")
    ax.plot(*ipt, "*", color="red", ms=18, zorder=6,
            label=f"Intercept ({miss_dist:.0f} m miss)")
    ax.plot(*LAUNCH_POS, "v", color=C["msle"], ms=12, label="Missile launch")
    ax.plot(*actual_pos, "X", color=C["true"], ms=12, zorder=6,
            label="Target at intercept")
    ax.add_patch(plt.Circle(ipt, KILL_RADIUS, color="red", fill=False,
                            ls="--", lw=1.5, alpha=0.7,
                            label=f"Kill radius ({KILL_RADIUS:.0f} m)"))
ax.plot(*ts[launch_step, :2], "^", color="purple", ms=10,
        label="Target at launch")
status = "HIT" if hit else "MISS"
ax.set_title(f"Plot 3: Intercept Geometry — {status} (representative mission)",
             color=("green" if hit else "red"), fontweight="bold")
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_aspect("equal")

# ── Plot 4: Multi-Mission Monte Carlo Performance ─────────────────────────────
ax = axes[1, 1]
labels = [f"Success Rate\n({hits}/{N_MISSIONS})", "Avg Error\nReduction %",
          "Avg Raw\nError (m)", "Avg KF\nError (m)"]
values = [success_pct, avg_err_pct, avg_raw_err, avg_kf_err]
bar_c  = ["#27ae60" if success_pct >= 80 else "#c0392b",
          "#3498db", "#e74c3c", "#2980b9"]
bars   = ax.bar(labels, values, color=bar_c, edgecolor="black", linewidth=0.8)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
            f"{v:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
ax.set_title(f"Plot 4: {N_MISSIONS}-Mission Monte Carlo Performance")
ax.set_ylabel("Value"); ax.grid(alpha=0.3, axis="y")
summary = (
    f"Missions run   : {N_MISSIONS}\n"
    f"Hits           : {hits}\n"
    f"Success rate   : {success_pct:.1f}%\n"
    f"Avg error red. : {avg_err_pct:.1f}%\n"
    f"Avg miss dist  : {avg_miss:.1f} m"
)
ax.text(0.97, 0.95, summary, transform=ax.transAxes, fontsize=9,
        va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

plt.tight_layout()
fig.savefig("results/kalman_missile_intercept.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Console Summary ───────────────────────────────────────────────────────────
print(f"\n{'='*52}")
print(f"  Missions Run      : {N_MISSIONS}")
print(f"  Hits              : {hits} / {N_MISSIONS}")
print(f"  Success Rate      : {success_pct:.1f}%")
print(f"  Avg Error Reduc.  : {avg_err_pct:.1f}%")
print(f"  Avg Raw Error     : {avg_raw_err:.1f} m")
print(f"  Avg Kalman Error  : {avg_kf_err:.1f} m")
print(f"  Avg Miss Distance : {avg_miss:.1f} m  (kill radius {KILL_RADIUS:.0f} m)")
print(f"  Plot saved to     : results/kalman_missile_intercept.png")
print(f"{'='*52}")
print("SIMULATION COMPLETE")
