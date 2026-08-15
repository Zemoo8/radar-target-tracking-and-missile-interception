# Theory and derivations

This document derives every equation used by the simulator. It is the companion
to the code: each section names the file and function that implements it.

- [1. Problem statement](#1-problem-statement)
- [2. Target motion model](#2-target-motion-model)
- [3. Process noise](#3-process-noise)
- [4. Measurement model](#4-measurement-model)
- [5. The Kalman filter](#5-the-kalman-filter)
- [6. Why the filter works here](#6-why-the-filter-works-here)
- [7. Filter consistency](#7-filter-consistency)
- [8. The intercept problem](#8-the-intercept-problem)
- [9. Guidance](#9-guidance)
- [10. The 3-D extension](#10-the-3-d-extension)
- [11. Parameter choices](#11-parameter-choices)
- [12. Computational cost](#12-computational-cost)

---

## 1. Problem statement

A radar observes a manoeuvring target at discrete instants separated by a step
$\Delta t$. Each return gives a noisy **position** only. We want, at every step:

1. the best estimate of the target's full kinematic state, including the
   **unmeasured** velocity;
2. a quantified uncertainty on that estimate;
3. a prediction far enough ahead to route an interceptor to a future rendezvous.

Writing the state as position stacked on velocity,

$$x_k = \begin{bmatrix} p_k \\ v_k \end{bmatrix} \in \mathbb{R}^{2n}, \qquad n \in \lbrace 2, 3 \rbrace$$

the problem is to compute the posterior $p(x_k \mid z_{1:k})$ recursively, in
constant time per step. Under linear dynamics and Gaussian noise that posterior
stays Gaussian, and propagating its mean and covariance *is* the Kalman filter.

---

## 2. Target motion model

The target is modelled as **nearly constant velocity** (NCV): it holds its
velocity, and the manoeuvre it performs is treated as an unknown random
acceleration acting over the step. Over one interval,

$$p_{k+1} = p_k + v_k \Delta t + \tfrac{1}{2} a_k \Delta t^2, \qquad v_{k+1} = v_k + a_k \Delta t$$

Setting the mean of $a_k$ to zero, the deterministic part of the transition is

$$F = \begin{bmatrix} I_n & \Delta t\, I_n \\ 0 & I_n \end{bmatrix}$$

so that $x_{k+1} = F x_k + w_k$, with $w_k$ capturing the acceleration term.

> **Implemented in** the matrix `F` of `kalman_filter()` in
> `kalman_missile_sim.py`, and `KF_F` in `radar_3d_sim.py`.

The *truth* generator uses the same kinematics but draws a fresh
$a_k \sim \mathcal{N}(0, q^2 I)$ at every step, so the target genuinely
manoeuvres. This matters: the filter's model is only an approximation of the
truth, which is the realistic case and the one where $Q$ has to do real work.

---

## 3. Process noise

Treat the acceleration as continuous white noise of intensity $q^2$. Its effect
accumulated over one step is

$$w_k = \int_{0}^{\Delta t} e^{A(\Delta t - \tau)} G\, a(\tau) \, d\tau, \qquad G = \begin{bmatrix} 0 \\ I_n \end{bmatrix}$$

Carrying out the integral for the double-integrator gives, per axis,

$$Q = q^2 \begin{bmatrix} \dfrac{\Delta t^4}{4} & \dfrac{\Delta t^3}{2} \\ \dfrac{\Delta t^3}{2} & \Delta t^2 \end{bmatrix}$$

Each entry has a physical reading:

| Entry | Origin | Meaning |
|---|---|---|
| $\Delta t^4/4$ | $(\tfrac{1}{2}a\Delta t^2)^2$ | position uncertainty injected by an unknown acceleration |
| $\Delta t^2$ | $(a\Delta t)^2$ | velocity uncertainty from the same cause |
| $\Delta t^3/2$ | cross term | position and velocity errors are **correlated**, because one acceleration causes both |

That off-diagonal term is the whole point. A hand-tuned diagonal $Q$ throws it
away and the filter then treats position and velocity errors as independent,
which makes it lag systematically through a turn. Keeping the correlation is
what lets the estimate bend with the target instead of trailing it.

> **Implemented in** `Q` in `kalman_missile_sim.py` and `KF_Q_MAT` in `radar_3d_sim.py`.

---

## 4. Measurement model

The radar returns position only:

$$z_k = H x_k + v_k, \qquad H = \begin{bmatrix} I_n & 0 \end{bmatrix}, \qquad v_k \sim \mathcal{N}(0, R)$$

$H$ has rank $n$ while the state has dimension $2n$: **velocity is not measured
at any single instant.** It is recovered only because the filter accumulates
information across time, which is the reason a filter is needed at all rather
than a per-frame smoother.

In 2-D, $R = \sigma_r^2 I_2$. In 3-D the noise is anisotropic,
$R = \mathrm{diag}(\sigma_r^2, \sigma_{alt}^2, \sigma_r^2)$, because altimetric
accuracy and horizontal accuracy come from different physical mechanisms.

---

## 5. The Kalman filter

### Predict

$$\hat{x}_{k|k-1} = F \hat{x}_{k-1|k-1}$$

$$P_{k|k-1} = F P_{k-1|k-1} F^{\top} + Q$$

Uncertainty always grows here: $F P F^{\top}$ spreads the existing uncertainty
through the dynamics, and $Q$ adds the manoeuvre we could not predict.

### Correct

Innovation, innovation covariance and gain:

$$y_k = z_k - H \hat{x}_{k|k-1}, \qquad S_k = H P_{k|k-1} H^{\top} + R, \qquad K_k = P_{k|k-1} H^{\top} S_k^{-1}$$

$$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k y_k, \qquad P_{k|k} = (I - K_k H) P_{k|k-1}$$

### Where the gain comes from

Write the estimation error after an update with an arbitrary gain $K$ and take
the trace of its covariance:

$$P_{k|k}(K) = (I - K H) P_{k|k-1} (I - K H)^{\top} + K R K^{\top}$$

Setting $\partial\, \mathrm{tr}\, P_{k|k} / \partial K = 0$ yields

$$-2 (I - K H) P_{k|k-1} H^{\top} + 2 K R = 0 \;\Longrightarrow\; K = P_{k|k-1} H^{\top} \left( H P_{k|k-1} H^{\top} + R \right)^{-1}$$

So the gain is not a tuning knob: it is the unique minimiser of the mean squared
error. It scales as "how uncertain am I" divided by "how uncertain am I plus how
noisy is the sensor". Confident prediction and noisy sensor gives a small gain;
uncertain prediction and clean sensor gives a gain approaching one.

---

## 6. Why the filter works here

The steady-state behaviour is governed by the ratio between the manoeuvre noise
and the measurement noise, often written

$$\lambda = \frac{q\, \Delta t^2}{\sigma_r}$$

With the study parameters ($q = 5$, $\Delta t = 0.5$, $\sigma_r = 50$) this is
small: the target is far more predictable than the sensor is accurate, so the
filter can average aggressively over many returns. That is precisely the regime
in which roughly half of the raw error is removed, and it is why the measured
50.7 % reduction is the expected order of magnitude rather than a surprise.

If $\lambda$ were large, the filter would have to trust each return almost
completely and the gain would approach identity, leaving the error essentially
unimproved. Reporting the regime alongside the number is what makes the number
mean something.

---

## 7. Filter consistency

Accuracy alone is not correctness. A filter is **consistent** when its reported
covariance actually describes its error. The two standard checks are the
normalised estimation error squared and the normalised innovation squared:

$$\mathrm{NEES}_k = (x_k - \hat{x}_{k|k})^{\top} P_{k|k}^{-1} (x_k - \hat{x}_{k|k}), \qquad \mathrm{NIS}_k = y_k^{\top} S_k^{-1} y_k$$

Under correct modelling these are chi-squared with $\dim(x)$ and $\dim(z)$
degrees of freedom. Values persistently above the bound mean the filter is
overconfident (typically $Q$ too small); values below mean it is throwing away
information. Adding these diagnostics is the first item on the roadmap in the
[README](../README.md).

---

## 8. The intercept problem

An interceptor at $p_I$ flying at constant speed $v_I$ must meet a target whose
predicted position at horizon $t$ is $\hat{p}(t) = \hat{p}_0 + \hat{v}_0 t$.
Rendezvous requires the interceptor to cover the separation in exactly the time
available:

$$\lVert \hat{p}_0 + \hat{v}_0 t - p_I \rVert = v_I t$$

Squaring gives a quadratic in $t$:

$$\underbrace{\left( \lVert \hat{v}_0 \rVert^2 - v_I^2 \right)}_{a} t^2 + \underbrace{2 \, d_0 \cdot \hat{v}_0}_{b} \, t + \underbrace{\lVert d_0 \rVert^2}_{c} = 0, \qquad d_0 = \hat{p}_0 - p_I$$

The smallest positive root is the earliest feasible intercept time. Since
$v_I > \lVert \hat{v}_0 \rVert$ in this scenario, $a < 0$ and a positive root
always exists: a faster pursuer can always catch a straight-flying target.

The implementation solves the equivalent **feasibility** form

$$t^{*} = \min \lbrace t > 0 : \lVert \hat{p}(t) - p_I \rVert \le v_I t \rbrace$$

by scanning forward on the simulation grid. This is deliberate: the scan
degrades gracefully when no solution exists (it simply reports failure instead
of returning a complex root), it extends unchanged to a non-linear predicted
trajectory, and it costs nothing at these horizon lengths.

> **Implemented in** `find_intercept()` in `kalman_missile_sim.py`.

---

## 9. Guidance

Committing to the launch-time solution would waste every measurement taken after
launch. The real-time simulator instead re-solves continuously with a shrinking
lead horizon $L_k$ and smooths the commanded aim point:

$$p^{cmd}_k = \hat{p}_k + \hat{v}_k L_k, \qquad m^{tgt}_k = (1 - \alpha)\, m^{tgt}_{k-1} + \alpha\, p^{cmd}_k$$

with $\alpha = 0.05$. The smoothing matters for a concrete reason: the
innovation sequence is white noise, so an unsmoothed command channel would
forward sensor noise directly into the steering demand and produce chattering.
The low-pass acts as the cheapest possible substitute for the actuator dynamics
that a real airframe would impose anyway.

The interceptor then advances along the unit vector toward the smoothed aim
point, and terminal success is declared when the separation falls below the
lethal radius.

---

## 10. The 3-D extension

The real-time simulator carries six states, $[p_x, p_y, p_z, v_x, v_y, v_z]$,
with

$$F = \begin{bmatrix} I_3 & \Delta t\, I_3 \\ 0 & I_3 \end{bmatrix}, \qquad H = \begin{bmatrix} I_3 & 0 \end{bmatrix}, \qquad R = \mathrm{diag}(\sigma_r^2, \sigma_{alt}^2, \sigma_r^2)$$

The structure is identical; only the dimensions change. That is the practical
virtue of the linear-Gaussian formulation, and it is why the same twenty lines
of algebra serve both the batch study and the interactive simulator.

The rendering pipeline is independent of the estimation. A camera basis is built
with a `look_at` construction, world points are mapped into camera coordinates,
culled against the near plane, and projected through a pinhole model:

$$s_x = \frac{c_x}{c_z} \cdot \frac{f}{\text{aspect}}, \qquad s_y = \frac{c_y}{c_z} \cdot f, \qquad f = \cot\left(\frac{\text{FOV}}{2}\right)$$

---

## 11. Parameter choices

| Parameter | Value | Why |
|---|---|---|
| $\Delta t = 0.5$ s | study | Fast enough to resolve a manoeuvre, slow enough to be a realistic revisit rate |
| $\sigma_r = 50$ m | study | Large relative to the per-step displacement, so the filtering problem is non-trivial |
| $q = 5$ m/s^2 | study | Roughly half a g: an evading aircraft, not a ballistic object |
| $P_0 = 5000 I$ | study | Deliberately pessimistic, so the filter must converge from a poor start rather than being handed the answer |
| lethal radius 300 m | study | Turns a continuous miss distance into a binary success criterion |
| 15 missions | study | Enough for a stable success rate and mean at negligible cost |

The initial covariance deserves a note. Starting at $P_0 = 5000 I$ with a
zero-velocity initial guess means the first estimates are badly wrong by
construction; the transient visible in the first seconds of panel 2 is the
filter converging, and including it in the average makes the reported
improvement conservative rather than flattering.

---

## 12. Computational cost

Per step the filter costs one $2n \times 2n$ propagation and one $n \times n$
inversion, i.e. $O(n^3)$ with $n \le 3$: constant time, constant memory,
independent of how long the track has been running. This is the structural
advantage of a recursive estimator over batch least squares, and the reason the
full 6-state update fits comfortably inside a 60 FPS frame budget alongside the
software renderer.

---

## Further reading

See the reference list in the [README](../README.md#13-references). The two
closest sources for this material are Bar-Shalom, Li and Kirubarajan for the
estimation theory and discretised process-noise models, and Zarchan for the
intercept and guidance geometry.
