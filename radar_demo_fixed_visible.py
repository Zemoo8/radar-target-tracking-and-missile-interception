"""
Radar Tracking 3D Demo
Python/Pygame port of radar_3d_demo_equations_v2.html

Requirements: pip install pygame-ce numpy   (or: pip install pygame numpy)
Controls: SPACE=pause  R=reset  N=noise  K=kalman  I=interceptor  +/-=speed  ESC=quit
"""

import pygame
import numpy as np
import math, random, sys

# ── Window ────────────────────────────────────────────────────────────────────
W, H = 1280, 720
pygame.init()
screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
pygame.display.set_caption("Radar Tracking 3D Demo")
clock  = pygame.time.Clock()
FPS    = 60

# ── Color palette ─────────────────────────────────────────────────────────────
BG        = (  2,   8,  16)
C_GREEN   = (  0, 255, 136)
C_GREEN_D = (  0,  80,  40)
C_BLUE    = (  0, 170, 255)
C_ORANGE  = (255, 170,   0)
C_RED     = (255,  50,  50)
C_CYAN    = (  0, 255, 204)
C_YELLOW  = (255, 220,   0)
C_DIM     = ( 74, 102,  85)
C_TRAIL_T = (  0, 240,  90)
C_TRAIL_K = (  0, 160, 255)
C_TRAIL_M = (255, 170,   0)
C_NOISE   = (255,  50,  50)
C_KFMARK  = (  0, 200, 255)
C_GRID    = (  0,  55,  28)
C_RING    = (  0,  80,  40)
C_TOWER   = ( 50, 110,  70)
C_PANEL   = (  0,  12,   6)

# ── Fonts ─────────────────────────────────────────────────────────────────────
def _font(sz):
    for name in ("Courier New", "Courier", "Lucida Console", "monospace"):
        try:
            f = pygame.font.SysFont(name, sz)
            if f:
                return f
        except Exception:
            pass
    return pygame.font.Font(None, sz + 4)

F_SM    = _font(11)
F_MD    = _font(13)
F_LG    = _font(16)
F_XL    = _font(20)
F_EQ    = _font(21)
F_PHASE = _font(14)

# ── Simulation constants (identical to HTML) ──────────────────────────────────
SIGMA_R   = 45.0
SIGMA_ALT = 18.0
KILL_R    = 90.0
T_ALT     = 210.0
T_SPD     = 4.3
PULSE_IV  = 55
LAUNCH_AT = 200
MS_SPEED  = 10.0
KF_Q_SIG  = 2.0
DT_KF     = 1.0

T_START    = np.array([-1100.0,  T_ALT, -680.0])
LAUNCH_POS = np.array([  180.0,    0.0,  180.0])

# ── Kalman matrices ───────────────────────────────────────────────────────────
KF_F = np.eye(6)
KF_F[0, 3] = KF_F[1, 4] = KF_F[2, 5] = DT_KF

_dt, _q2 = DT_KF, KF_Q_SIG ** 2
KF_Q_MAT = np.array([
    [_dt**4/4, 0,        0,        _dt**3/2, 0,        0       ],
    [0,        _dt**4/4, 0,        0,        _dt**3/2, 0       ],
    [0,        0,        _dt**4/4, 0,        0,        _dt**3/2],
    [_dt**3/2, 0,        0,        _dt**2,   0,        0       ],
    [0,        _dt**3/2, 0,        0,        _dt**2,   0       ],
    [0,        0,        _dt**3/2, 0,        0,        _dt**2  ],
]) * _q2

KF_H = np.zeros((3, 6))
KF_H[0, 0] = KF_H[1, 1] = KF_H[2, 2] = 1.0


def kf_step(x, P, z):
    R  = np.diag([SIGMA_R**2, SIGMA_ALT**2, SIGMA_R**2])
    xp = KF_F @ x
    Pp = KF_F @ P @ KF_F.T + KF_Q_MAT
    S  = KF_H @ Pp @ KF_H.T + R
    K  = Pp @ KF_H.T @ np.linalg.inv(S)
    xn = xp + K @ (z - KF_H @ xp)
    Pn = (np.eye(6) - K @ KF_H) @ Pp
    return xn, Pn


# ── 3D Camera / Projection ────────────────────────────────────────────────────
# Look-at matrix: rows = [right, up, -forward]
# In camera space:  cp[2] = (-fwd) . rel  →  depth = -cp[2] = fwd . rel
# Points in front:  fwd . rel > 0  →  depth > 0
FOV_DEG = 52.0
_FOV_F  = 1.0 / math.tan(math.radians(FOV_DEG / 2))   # ≈ 2.05


def look_at(pos, target, up=np.array([0., 1., 0.])):
    fwd = target - pos;  fwd /= np.linalg.norm(fwd)
    rgt = np.cross(fwd, up);  rgt /= np.linalg.norm(rgt)
    u   = np.cross(rgt, fwd)
    return np.array([rgt, u, -fwd])          # 3×3 view rotation matrix


def _proj_cp(cp, sw, sh):
    """Convert already-transformed camera-space point to screen coords."""
    depth = -cp[2]                           # ← KEY FIX: depth = -cp[2]
    if depth <= 0.5:
        return None
    asp = sw / sh
    sx  = (cp[0] / depth) * _FOV_F / asp
    sy  = (cp[1] / depth) * _FOV_F
    px  = int((sx + 1.0) * 0.5 * sw)
    py  = int((1.0 - (sy + 1.0) * 0.5) * sh)
    return px, py, depth


def proj(world_pt, cam_pos, cam_rot):
    cp = cam_rot @ (np.asarray(world_pt, float) - cam_pos)
    return _proj_cp(cp, W, H)


def proj_batch(pts, cam_pos, cam_rot):
    """Vectorised: returns list of (px,py,depth) or None per point."""
    if not pts:
        return []
    arr   = np.asarray(pts, float)
    cp    = (cam_rot @ (arr - cam_pos).T).T          # (N,3)
    depth = -cp[:, 2]                                # (N,)  ← KEY FIX
    valid = depth > 0.5
    asp   = W / H
    # Safe divide: only divide where valid
    d_safe = np.where(valid, depth, 1.0)
    sx     = (cp[:, 0] / d_safe) * _FOV_F / asp
    sy     = (cp[:, 1] / d_safe) * _FOV_F
    px     = ((sx + 1.0) * 0.5 * W).astype(int)
    py     = ((1.0 - (sy + 1.0) * 0.5) * H).astype(int)
    return [(int(px[i]), int(py[i]), float(depth[i])) if valid[i] else None
            for i in range(len(pts))]


def draw_line3d(surf, p1, p2, cam_pos, cam_rot, color, w=1):
    r1 = proj(p1, cam_pos, cam_rot)
    r2 = proj(p2, cam_pos, cam_rot)
    if r1 is None or r2 is None:
        return
    try:
        pygame.draw.line(surf, color, (r1[0], r1[1]), (r2[0], r2[1]), w)
    except Exception:
        pass


def draw_dot3d(surf, pt, cam_pos, cam_rot, color, radius=3):
    r = proj(pt, cam_pos, cam_rot)
    if r is None:
        return
    x, y, _ = r
    if -radius*2 <= x <= W + radius*2 and -radius*2 <= y <= H + radius*2:
        try:
            pygame.draw.circle(surf, color, (x, y), radius)
        except Exception:
            pass


def draw_trail(surf, pts, cam_pos, cam_rot, color, w=2):
    if len(pts) < 2:
        return
    projected = proj_batch(pts, cam_pos, cam_rot)
    for i in range(len(projected) - 1):
        a, b = projected[i], projected[i + 1]
        if a is None or b is None:
            continue
        try:
            pygame.draw.line(surf, color, (a[0], a[1]), (b[0], b[1]), w)
        except Exception:
            pass


# ── Pre-computed static geometry ──────────────────────────────────────────────
random.seed(42)
STARS = []
for _ in range(500):
    r   = 4500 + random.random() * 2000
    th  = random.random() * math.pi * 2
    phi = random.random() * math.pi * 0.48
    STARS.append(np.array([r * math.sin(phi) * math.cos(th),
                            r * math.cos(phi) + 200,
                            r * math.sin(phi) * math.sin(th)]))
random.seed()

# Ground grid (every 200 units, ±2000)
GRID_LINES = []
for i in range(-2000, 2001, 200):
    GRID_LINES.append((np.array([float(i), 0., -2000.]),
                       np.array([float(i), 0.,  2000.])))
    GRID_LINES.append((np.array([-2000., 0., float(i)]),
                       np.array([ 2000., 0., float(i)])))

# Radar range rings
RANGE_RINGS = []
for rad in (400, 800, 1200):
    pts = []
    for a in range(0, 361, 10):
        ang = math.radians(a)
        pts.append(np.array([rad * math.cos(ang), 0., rad * math.sin(ang)]))
    RANGE_RINGS.append(pts)


def _cyl(cx, cz, ybot, r_bot, r_top, height, segs=8):
    lines = []
    angs  = [math.radians(i * 360 / segs) for i in range(segs)]
    bot   = [np.array([cx + r_bot*math.cos(a), ybot,          cz + r_bot*math.sin(a)]) for a in angs]
    top   = [np.array([cx + r_top*math.cos(a), ybot+height,   cz + r_top*math.sin(a)]) for a in angs]
    for i in range(segs):
        lines += [(bot[i], bot[(i+1)%segs]),
                  (top[i], top[(i+1)%segs]),
                  (bot[i], top[i])]
    return lines


TOWER_LINES = (
    _cyl(0, 0,  0, 52, 40,  8, 12) +   # platform
    _cyl(0, 0, 12, 10,  6, 72,  8)      # tower
)
for a in range(0, 361, 18):             # radar dish ring at y=78
    a1, a2 = math.radians(a), math.radians(a + 18)
    TOWER_LINES.append((np.array([28*math.cos(a1), 78., 28*math.sin(a1)]),
                        np.array([28*math.cos(a2), 78., 28*math.sin(a2)])))

PAD_LINES = (
    _cyl(LAUNCH_POS[0], LAUNCH_POS[2], 0, 26, 22, 6, 8) +
    [(np.array([LAUNCH_POS[0],  6., LAUNCH_POS[2]]),
      np.array([LAUNCH_POS[0], 36., LAUNCH_POS[2]]))]
)

# ── Equation data (ASCII-safe) ────────────────────────────────────────────────
EQUATIONS = [
    dict(
        title="RADAR MEASUREMENT MODEL",
        eq="z_k = H x_k + v_k",
        expl=("The radar measures target position,\n"
              "but additive noise v_k corrupts the reading."),
        color=C_GREEN,
    ),
    dict(
        title="KALMAN CORRECTION STEP",
        eq="x_hat(k|k) = x_hat(k|k-1) + K_k (z_k - H x_hat(k|k-1))",
        expl=("The filter corrects the prediction\n"
              "using the noisy radar measurement."),
        color=C_BLUE,
    ),
    dict(
        title="INTERCEPT OPTIMISATION",
        eq="||p_T(tau+t) - p_I|| <= v_I * t",
        expl=("Search for the earliest future point\n"
              "the interceptor can physically reach."),
        color=C_ORANGE,
    ),
    dict(
        title="GUIDANCE UPDATE",
        eq="p_T(tau+t) ~= p_hat(tau) + v_hat(tau) * t",
        expl=("Estimated position and velocity predict\n"
              "where the target will be at intercept."),
        color=C_BLUE,
    ),
]

# ── Simulation state ──────────────────────────────────────────────────────────
class Sim:
    def reset(self):
        self.t          = 0
        self.orbit_t    = 0.0
        self.paused     = False
        self.speed      = 1

        self.tPos = T_START.copy().astype(float)
        v0        = np.array([T_SPD, 0., T_SPD * 0.75])
        self.tVel = v0 / np.linalg.norm(v0) * T_SPD
        self.tAcc = np.zeros(3)

        self.kfX = np.concatenate([self.tPos.copy(), self.tVel.copy()])
        self.kfP = np.diag([SIGMA_R**2]*3 + [300.]*3).astype(float)

        self.raw_errs   = []
        self.kf_errs    = []
        self.det_count  = 0
        self.pulse_t    = 0

        self.true_pts   = []
        self.kf_pts     = []
        self.m_trail    = []
        self.noise_dots = []

        self.m_launched  = False
        self.mPos        = np.array([LAUNCH_POS[0], 35., LAUNCH_POS[2]])
        self.mTgt        = np.zeros(3)
        self.m_time      = 0
        self.intercepted = False
        self.exploding   = False
        self.expl_t      = 0
        self.expl_parts  = []
        self.pulse_rings = []

        self.layers = {'noise': True, 'kalman': True, 'interceptor': True}

        self.phase_text  = 'RADAR ACQUISITION'
        self.phase_color = C_GREEN
        self.mode_text   = 'ACQUISITION'
        self.mode_color  = C_CYAN
        self.radar_text  = 'SCANNING'
        self.mis_status  = 'STANDBY'
        self.tti_text    = '--'

        # Equation chain
        self.eq_started  = False
        self.eq_active   = False
        self.eq_idx      = 0
        self.eq_chars    = 0
        self.eq_type_t   = 0.0
        self.eq_timer    = 0.0
        self.cinematic   = False

        # Calc flash
        self.flash_text  = ''
        self.flash_t     = 0.0

        self.reset_at    = 0

    def __init__(self):
        self.reset()


sim = Sim()


def _start_eq(idx):
    sim.eq_idx    = idx
    sim.eq_active = True
    sim.eq_chars  = 0
    sim.eq_type_t = 0.0
    sim.eq_timer  = 0.0
    sim.cinematic = True


def _advance_eq():
    sim.eq_active = False
    nxt = sim.eq_idx + 1
    if nxt < len(EQUATIONS):
        _start_eq(nxt)
    else:
        sim.cinematic = False


# ── Simulation step (identical logic to HTML) ─────────────────────────────────
def sim_step():
    if sim.intercepted:
        return
    sim.t += 1

    # Target movement
    if sim.t % 18 == 0:
        sim.tAcc = np.array([(random.random()-.5)*1.9,
                              (random.random()-.5)*.09,
                              (random.random()-.5)*1.9])
    sim.tVel += sim.tAcc
    spd = float(np.linalg.norm(sim.tVel))
    if spd > T_SPD * 1.4: sim.tVel *= (T_SPD * 1.4) / spd
    if spd < T_SPD * 0.7: sim.tVel = sim.tVel / spd * (T_SPD * 0.7)
    sim.tVel[1] += (T_ALT - sim.tPos[1]) * 0.009
    sim.tVel[1]  = max(-1.3, min(1.3, sim.tVel[1]))
    sim.tPos    += sim.tVel
    sim.tPos[0]  = max(-1350., min(1350., sim.tPos[0]))
    sim.tPos[2]  = max(-1350., min(1350., sim.tPos[2]))

    sim.true_pts.append(sim.tPos.copy())
    if len(sim.true_pts) > 260:
        sim.true_pts.pop(0)

    # Radar ping
    sim.pulse_t += 1
    if sim.pulse_t >= PULSE_IV:
        sim.pulse_t = 0
        sim.pulse_rings.append({'r': 0., 'max_r': 1600., 'spd': 9.})

        if not sim.eq_started:
            sim.eq_started = True
            _start_eq(0)

        z = np.array([
            sim.tPos[0] + (random.random()-.5) * SIGMA_R   * 2.3,
            sim.tPos[1] + (random.random()-.5) * SIGMA_ALT * 2.3,
            sim.tPos[2] + (random.random()-.5) * SIGMA_R   * 2.3,
        ])

        if len(sim.noise_dots) > 55:
            sim.noise_dots.pop(0)
        sim.noise_dots.append(z.copy())

        sim.raw_errs.append(float(np.linalg.norm(z - sim.tPos)))
        sim.kfX, sim.kfP = kf_step(sim.kfX, sim.kfP, z)
        sim.kf_errs.append(float(np.linalg.norm(sim.kfX[:3] - sim.tPos)))

        sim.kf_pts.append(sim.kfX[:3].copy())
        if len(sim.kf_pts) > 260:
            sim.kf_pts.pop(0)
        sim.det_count += 1

    # Pulse ring expansion
    for ring in sim.pulse_rings[:]:
        ring['r'] += ring['spd']
        if ring['r'] >= ring['max_r']:
            sim.pulse_rings.remove(ring)

    # CALCULATING flash
    if sim.t == LAUNCH_AT - 55:
        sim.flash_text = 'CALCULATING...'
        sim.flash_t    = 0.7

    # Interceptor launch
    if sim.t == LAUNCH_AT and not sim.m_launched:
        sim.m_launched = True
        lead    = 95
        sim.mTgt = np.array([
            sim.kfX[0] + sim.kfX[3] * lead,
            max(sim.kfX[1] + sim.kfX[4] * lead, 40.),
            sim.kfX[2] + sim.kfX[5] * lead,
        ])
        sim.mPos        = np.array([LAUNCH_POS[0], 35., LAUNCH_POS[2]])
        sim.m_time      = 0
        sim.phase_text  = 'INTERCEPTOR LAUNCHED'
        sim.phase_color = C_ORANGE
        sim.mode_text   = 'ENGAGING'
        sim.mode_color  = C_RED
        sim.flash_text  = 'SOLUTION FOUND'
        sim.flash_t     = 0.8

    # Interceptor guidance
    if sim.m_launched and not sim.intercepted:
        sim.m_time += 1
        lead = max(0, 95 - sim.m_time)
        pred = np.array([
            sim.kfX[0] + sim.kfX[3] * lead,
            max(sim.kfX[1] + sim.kfX[4] * lead, 40.),
            sim.kfX[2] + sim.kfX[5] * lead,
        ])
        sim.mTgt = sim.mTgt * 0.95 + pred * 0.05

        d    = sim.mTgt - sim.mPos
        dist = float(np.linalg.norm(d))
        if dist > 0:
            d /= dist
        sim.mPos += d * min(dist, MS_SPEED)

        sim.m_trail.append(sim.mPos.copy())
        if len(sim.m_trail) > 130:
            sim.m_trail.pop(0)

        miss_dist = float(np.linalg.norm(sim.mPos - sim.tPos))
        tti = miss_dist / (MS_SPEED * 60.)
        sim.tti_text   = f"{tti:.1f} s"
        sim.mis_status = 'IN FLIGHT'
        sim.radar_text = 'TRACKING + GUIDING'

        if miss_dist < KILL_R or sim.m_time > 300:
            sim.intercepted  = True
            _spawn_explosion(sim.mPos.copy())
            sim.phase_text   = 'INTERCEPT COMPLETE'
            sim.phase_color  = C_RED
            sim.mode_text    = 'DESTROYED'
            sim.mode_color   = C_RED
            sim.mis_status   = 'INTERCEPT OK'
            sim.tti_text     = '--'
            sim.reset_at     = pygame.time.get_ticks() + 4500

    # Status text while hunting
    if not sim.m_launched:
        if sim.t > 30:
            sim.mode_text  = 'TRACKING'
            sim.mode_color = C_ORANGE
        sim.radar_text = 'PINGING' if sim.pulse_t < 8 else 'SCANNING'
        sim.mis_status = 'STANDBY'
        sim.tti_text   = '--'


def _spawn_explosion(pos):
    sim.exploding = True
    sim.expl_t    = 0
    sim.expl_parts.clear()
    pal = [(255,85,0),(255,140,0),(255,210,0),(255,255,200),(255,34,0),(255,230,0)]
    for i in range(110):
        spd = random.random() * 5.5 + 0.8
        th  = random.random() * math.pi * 2
        phi = random.random() * math.pi
        sim.expl_parts.append({
            'pos':   pos.copy(),
            'vel':   np.array([math.sin(phi)*math.cos(th)*spd,
                               math.cos(phi)*spd*0.75,
                               math.sin(phi)*math.sin(th)*spd]),
            'life':  1.0,
            'dr':    random.random()*0.016 + 0.007,
            'color': pal[i % len(pal)],
            'r':     random.randint(1, 4),
            'flash': False,
        })
    sim.expl_parts.append({
        'pos': pos.copy(), 'vel': None,
        'life': 1.0, 'dr': 0.038,
        'color': (255, 220, 100), 'r': 8, 'flash': True,
    })


# ── HUD helpers ───────────────────────────────────────────────────────────────
def _panel(surf, x, y, w, h, border=C_GREEN):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill((0, 12, 6, 210))
    pygame.draw.rect(s, (*border, 60), (0, 0, w, h), 1)
    surf.blit(s, (x, y))


def _txt(surf, text, font, color, x, y):
    surf.blit(font.render(text, True, color), (x, y))
    return font.size(text)[1]


def _row(surf, lbl, val, lc, vc, x, y, x_end):
    ls = F_SM.render(lbl, True, lc)
    vs = F_SM.render(val, True, vc)
    surf.blit(ls, (x, y))
    surf.blit(vs, (x_end - vs.get_width(), y))
    return ls.get_height() + 3


def _div(surf, x, y, w):
    s = pygame.Surface((w, 1), pygame.SRCALPHA)
    s.fill((0, 255, 136, 20))
    surf.blit(s, (x, y))


def _kf_conf():
    if not sim.raw_errs:
        return '--', C_GREEN
    n  = min(8, len(sim.raw_errs))
    ar = sum(sim.raw_errs[-n:]) / n
    ak = sum(sim.kf_errs[-n:]) / n
    r  = max(0., (1 - ak/ar)*100) if ar > 0 else 0.
    if r > 35: return 'HIGH',   C_CYAN
    if r > 12: return 'MEDIUM', C_GREEN
    return 'LOW', C_ORANGE


def _errs():
    if not sim.raw_errs:
        return 0., 0., 0.
    n  = min(8, len(sim.raw_errs))
    ar = sum(sim.raw_errs[-n:]) / n
    ak = sum(sim.kf_errs[-n:]) / n
    return ar, ak, max(0., (1 - ak/ar)*100) if ar > 0 else 0.


# ── HUD drawing ───────────────────────────────────────────────────────────────
def draw_hud(surf):
    sw, sh = surf.get_size()

    # Corner brackets
    bl = 28
    for cx_, cy_, dx, dy in [(0,0,1,1),(sw,0,-1,1),(0,sh,1,-1),(sw,sh,-1,-1)]:
        pygame.draw.line(surf, C_GREEN_D, (cx_, cy_), (cx_+dx*bl, cy_), 2)
        pygame.draw.line(surf, C_GREEN_D, (cx_, cy_), (cx_, cy_+dy*bl), 2)

    # ── System Status (top-left) ──────────────────────────────────────────────
    pw, px, py = 218, 14, 14
    _panel(surf, px, py, pw, 170)
    cx, cy = px+11, py+9
    xe = px + pw - 11
    _txt(surf, '>> SYSTEM STATUS', F_SM, (0,200,110), cx, cy);  cy += 16
    _div(surf, cx, cy-2, pw-22);  cy += 2
    kc, kcc = _kf_conf()
    cy += _row(surf, 'MODE',       sim.mode_text,      C_DIM, sim.mode_color, cx, cy, xe)
    cy += _row(surf, 'RADAR',      sim.radar_text,     C_DIM, C_GREEN,        cx, cy, xe)
    cy += _row(surf, 'KF CONF',    kc,                 C_DIM, kcc,            cx, cy, xe)
    cy += _row(surf, 'DETECTIONS', str(sim.det_count), C_DIM, C_GREEN,        cx, cy, xe)
    cy += 4
    surf.blit(F_SM.render('MISSION PROGRESS', True, C_DIM), (cx, cy))
    cy += F_SM.get_height() + 3
    bw = pw - 22
    pygame.draw.rect(surf, (0, 255, 136, 14), (cx, cy, bw, 5))
    pygame.draw.rect(surf, (0, 255, 136, 28), (cx, cy, bw, 5), 1)
    fill = int(bw * min(1., sim.t / (LAUNCH_AT + 240)))
    if fill > 0:
        pygame.draw.rect(surf, C_GREEN, (cx, cy, fill, 5))

    # ── Phase banner (top-centre) ─────────────────────────────────────────────
    show = True
    if sim.phase_text in ('RADAR ACQUISITION', 'INTERCEPTOR LAUNCHED'):
        show = (pygame.time.get_ticks() // 450) % 2 == 0
    if show:
        ps  = F_PHASE.render(sim.phase_text, True, sim.phase_color)
        bw2 = ps.get_width() + 44
        bpx = sw//2 - bw2//2
        _panel(surf, bpx, 14, bw2, 36)
        surf.blit(ps, (sw//2 - ps.get_width()//2, 22))

    # ── Live Tracking Metrics (top-right) ─────────────────────────────────────
    mpw = 248;  mpx = sw - mpw - 14
    _panel(surf, mpx, 14, mpw, 190)
    cx2, cy2 = mpx+11, 14+9
    xe2 = mpx + mpw - 11
    _txt(surf, '<> LIVE TRACKING METRICS', F_SM, (0,200,110), cx2, cy2);  cy2 += 16
    _div(surf, cx2, cy2-2, mpw-22);  cy2 += 2
    rng = math.sqrt(sim.tPos[0]**2 + sim.tPos[2]**2)
    spd = float(np.linalg.norm(sim.tVel)) * 60
    cy2 += _row(surf, 'Target Range', f'{rng:.0f} m',         C_DIM, C_GREEN,  cx2, cy2, xe2)
    cy2 += _row(surf, 'Altitude',     f'{sim.tPos[1]:.0f} m', C_DIM, C_GREEN,  cx2, cy2, xe2)
    cy2 += _row(surf, 'Speed',        f'{spd:.0f} m/s',       C_DIM, C_GREEN,  cx2, cy2, xe2)
    _div(surf, cx2, cy2+1, mpw-22);  cy2 += 7
    ar, ak, red = _errs()
    cy2 += _row(surf, 'Raw Radar Error',   f'{ar:.1f} m' if ar else '--', C_DIM, C_ORANGE, cx2, cy2, xe2)
    cy2 += _row(surf, 'KF Position Error', f'{ak:.1f} m' if ak else '--', C_DIM, C_BLUE,   cx2, cy2, xe2)
    cy2 += _row(surf, 'Error Reduction',   f'{red:.1f}%' if ar else '--', C_DIM, C_CYAN,   cx2, cy2, xe2)
    _div(surf, cx2, cy2+1, mpw-22);  cy2 += 7
    mc = C_CYAN if sim.mis_status == 'STANDBY' else (C_GREEN if 'OK' in sim.mis_status else C_ORANGE)
    cy2 += _row(surf, 'Interceptor Status', sim.mis_status, C_DIM, mc,    cx2, cy2, xe2)
    cy2 += _row(surf, 'Time to Impact',     sim.tti_text,   C_DIM, C_RED, cx2, cy2, xe2)

    # ── Legend (bottom-right) ─────────────────────────────────────────────────
    lpw = 175;  lpx = sw - lpw - 14;  lpy = sh - 102
    _panel(surf, lpx, lpy, lpw, 102)
    lx, ly = lpx+10, lpy+8
    _txt(surf, '(o) LEGEND', F_SM, (0,200,110), lx, ly);  ly += 16
    for kind, col, lbl in [
        ('line', C_TRAIL_T, 'True trajectory'),
        ('dot',  C_NOISE,   'Radar noise'),
        ('line', C_TRAIL_K, 'Kalman estimate'),
        ('dot',  C_KFMARK,  'KF position'),
        ('line', C_TRAIL_M, 'Interceptor trail'),
    ]:
        if kind == 'line':
            pygame.draw.line(surf, col, (lx, ly+5), (lx+16, ly+5), 2)
        else:
            pygame.draw.circle(surf, col, (lx+5, ly+5), 4)
        _txt(surf, lbl, F_SM, C_DIM, lx+22, ly)
        ly += 15

    # ── Controls (bottom-centre) ──────────────────────────────────────────────
    cpw = 500;  cph = 38;  cpx = sw//2 - cpw//2;  cpy = sh - cph - 14
    _panel(surf, cpx, cpy, cpw, cph)
    bx, by = cpx+12, cpy+11
    play_t = '|| PAUSE' if not sim.paused else '> PLAY'
    play_c = C_CYAN if not sim.paused else C_GREEN
    _txt(surf, play_t, F_SM, play_c, bx, by);  bx += F_SM.size(play_t)[0] + 18
    _txt(surf, 'SPEED', F_SM, C_DIM, bx, by);  bx += F_SM.size('SPEED')[0] + 5
    _txt(surf, f'{sim.speed}x', F_SM, C_GREEN, bx, by); bx += F_SM.size(f'{sim.speed}x')[0] + 5
    _txt(surf, '[-/+]', F_SM, C_DIM, bx, by);  bx += F_SM.size('[-/+]')[0] + 18
    for name, key in [('NOISE','N'), ('KALMAN','K'), ('INTERCEPTOR','I')]:
        on  = sim.layers[name.lower()]
        col = C_GREEN if on else C_DIM
        lbl = f'[{key}]{name}'
        _txt(surf, lbl, F_SM, col, bx, by);  bx += F_SM.size(lbl)[0] + 12
    _txt(surf, '[R] RESET', F_SM, C_DIM, bx, by)


# ── Equation overlay ──────────────────────────────────────────────────────────
def draw_eq_overlay(surf):
    if not sim.eq_active:
        return
    sw, sh = surf.get_size()
    eq  = EQUATIONS[sim.eq_idx]
    col = eq['color']

    # Subtle dim — scene remains clearly visible
    dim = pygame.Surface((sw, sh), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 45))
    surf.blit(dim, (0, 0))

    epw, eph = 460, 158
    epx = 38;  epy = sh//2 - eph//2
    panel = pygame.Surface((epw, eph), pygame.SRCALPHA)
    panel.fill((0, 8, 4, 230))
    pygame.draw.rect(panel, (*col, 20), (0, 0, epw, eph), 1)
    pygame.draw.rect(panel, (*col, 255), (0, 0, 3, eph))   # left accent bar
    surf.blit(panel, (epx, epy))

    cx2, cy2 = epx+16, epy+11
    _txt(surf, 'MATHEMATICAL STEP — REPORT REFERENCE', F_SM, (0, 200, 110, 60), cx2, cy2)
    cy2 += 15
    _txt(surf, eq['title'], F_SM, col, cx2, cy2);  cy2 += 18
    _div(surf, cx2, cy2, epw-32);  cy2 += 9
    _txt(surf, eq['eq'][:sim.eq_chars], F_EQ, col, cx2, cy2);  cy2 += 34
    if sim.eq_chars >= len(eq['eq']):
        for line in eq['expl'].split('\n'):
            _txt(surf, line, F_SM, C_DIM, cx2, cy2);  cy2 += 15


def draw_calc_flash(surf):
    if not sim.flash_text:
        return
    sw, sh = surf.get_size()
    col = C_ORANGE if 'CALC' in sim.flash_text else C_CYAN
    s   = F_XL.render(sim.flash_text, True, col)
    surf.blit(s, (sw//2 - s.get_width()//2, sh//2 - s.get_height()//2))


# ── 3D Scene ──────────────────────────────────────────────────────────────────
def draw_scene(surf, cam_pos, cam_rot):
    # Stars
    for pt in STARS:
        r = proj(pt, cam_pos, cam_rot)
        if r and 0 <= r[0] <= W and 0 <= r[1] <= H:
            pygame.draw.circle(surf, (110, 155, 185), (r[0], r[1]), 1)

    # Ground grid
    for p1, p2 in GRID_LINES:
        draw_line3d(surf, p1, p2, cam_pos, cam_rot, C_GRID)

    # Radar range rings
    for ring_pts in RANGE_RINGS:
        for i in range(len(ring_pts)-1):
            draw_line3d(surf, ring_pts[i], ring_pts[i+1], cam_pos, cam_rot, C_RING)

    # Expanding radar pulse rings (on ground plane)
    if sim.layers['noise']:
        for ring in sim.pulse_rings:
            alpha = 0.88 * (1. - ring['r'] / ring['max_r'])
            cv    = int(alpha * 200)
            if cv > 4:
                for a in range(0, 360, 10):
                    a1 = math.radians(a);   a2 = math.radians(a + 10)
                    p1 = np.array([ring['r']*math.cos(a1), 0., ring['r']*math.sin(a1)])
                    p2 = np.array([ring['r']*math.cos(a2), 0., ring['r']*math.sin(a2)])
                    draw_line3d(surf, p1, p2, cam_pos, cam_rot, (0, cv, cv//2))

    # Radar tower
    for p1, p2 in TOWER_LINES:
        draw_line3d(surf, p1, p2, cam_pos, cam_rot, C_TOWER)

    # Sweeping radar beam
    dish_a = (sim.t * 0.024) % (math.pi * 2)
    bx_    =  1400 * math.cos(dish_a)
    bz_    =  1400 * math.sin(dish_a)
    draw_line3d(surf, np.array([0., 78., 0.]),
                np.array([bx_, 118., bz_]),
                cam_pos, cam_rot, (0, 80, 45))

    # Launch pad
    for p1, p2 in PAD_LINES:
        draw_line3d(surf, p1, p2, cam_pos, cam_rot, (55, 90, 45))

    # True trajectory (green)
    if sim.layers['kalman'] and len(sim.true_pts) > 1:
        draw_trail(surf, sim.true_pts, cam_pos, cam_rot, C_TRAIL_T, 2)

    # Radar noise dots (red)
    if sim.layers['noise']:
        for pt in sim.noise_dots:
            draw_dot3d(surf, pt, cam_pos, cam_rot, C_NOISE, 3)

    # Kalman estimate trail (blue)
    if sim.layers['kalman'] and len(sim.kf_pts) > 1:
        draw_trail(surf, sim.kf_pts, cam_pos, cam_rot, C_TRAIL_K, 2)

    # KF position marker (wireframe box)
    if sim.layers['kalman'] and sim.kf_pts:
        kfp = sim.kfX[:3]
        s   = 9
        corners = [np.array([kfp[0]+dx, kfp[1]+dy, kfp[2]+dz])
                   for dx in (-s, s) for dy in (-s, s) for dz in (-s, s)]
        edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),
                 (4,5),(4,6),(5,7),(6,7)]
        for i, j in edges:
            draw_line3d(surf, corners[i], corners[j], cam_pos, cam_rot, C_KFMARK)

    # Target aircraft (visible until intercept)
    if not sim.intercepted:
        p   = sim.tPos
        fwd = sim.tVel / max(float(np.linalg.norm(sim.tVel)), 1e-4)
        rgt = np.cross(fwd, np.array([0., 1., 0.]))
        n   = float(np.linalg.norm(rgt))
        rgt = rgt / n if n > 1e-4 else np.array([0., 0., 1.])
        up_ = np.cross(rgt, fwd)
        # Fuselage
        draw_line3d(surf, p - fwd*31, p + fwd*59, cam_pos, cam_rot, (70, 130, 230), 2)
        # Wings
        draw_line3d(surf, p - rgt*36, p + rgt*36, cam_pos, cam_rot, (55, 110, 200), 2)
        # Tail fin
        draw_line3d(surf, p - fwd*22 - up_*5, p - fwd*22 + up_*16,
                    cam_pos, cam_rot, (55, 110, 200), 2)
        # Engine glow
        for z_off in (-1, 1):
            draw_dot3d(surf, p - fwd*30 + rgt*22*z_off,
                       cam_pos, cam_rot, (255, 110, 0), 4)

    # Interceptor missile + trail
    if sim.layers['interceptor']:
        draw_trail(surf, sim.m_trail, cam_pos, cam_rot, C_TRAIL_M, 2)
        if sim.m_launched and not sim.intercepted:
            md  = sim.mTgt - sim.mPos
            nmd = float(np.linalg.norm(md))
            md  = md / nmd if nmd > 1e-4 else np.array([0., 1., 0.])
            draw_line3d(surf, sim.mPos - md*17, sim.mPos + md*17,
                        cam_pos, cam_rot, C_YELLOW, 3)
            draw_dot3d(surf, sim.mPos - md*17, cam_pos, cam_rot, (255, 110, 0), 5)

    # Explosion (drawn on top regardless of intercepted state)
    if sim.exploding:
        for part in sim.expl_parts:
            if part['life'] <= 0:
                continue
            if part['flash']:
                sc = max(4, int(part['r'] * (2.0 - part['life'])))
                draw_dot3d(surf, part['pos'], cam_pos, cam_rot, part['color'], sc)
            else:
                draw_dot3d(surf, part['pos'], cam_pos, cam_rot, part['color'], part['r'])


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    global W, H, screen

    while True:
        dt_ms  = clock.tick(FPS)
        dt_sec = dt_ms / 1000.

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if   k == pygame.K_ESCAPE: pygame.quit(); sys.exit()
                elif k == pygame.K_SPACE:  sim.paused = not sim.paused
                elif k == pygame.K_r:      sim.reset()
                elif k == pygame.K_n:      sim.layers['noise']       = not sim.layers['noise']
                elif k == pygame.K_k:      sim.layers['kalman']      = not sim.layers['kalman']
                elif k == pygame.K_i:      sim.layers['interceptor'] = not sim.layers['interceptor']
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    sim.speed = min(4, sim.speed + 1)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    sim.speed = max(1, sim.speed - 1)
            elif event.type == pygame.VIDEORESIZE:
                W, H   = event.w, event.h
                screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)

        # Auto-reset after intercept
        if sim.intercepted and sim.reset_at and pygame.time.get_ticks() >= sim.reset_at:
            sim.reset()

        # Simulation steps
        if not sim.paused:
            steps = 1 if sim.cinematic else max(1, sim.speed)
            for _ in range(steps):
                sim_step()
            sim.orbit_t += 0.00018 * steps

        # Calc flash timer
        if sim.flash_t > 0:
            sim.flash_t -= dt_sec
            if sim.flash_t <= 0:
                sim.flash_text = ''

        # Equation typewriter + auto-advance
        if sim.eq_active:
            sim.eq_type_t += dt_sec
            sim.eq_chars   = min(len(EQUATIONS[sim.eq_idx]['eq']),
                                  int(sim.eq_type_t * 18))
            sim.eq_timer  += dt_sec
            if sim.eq_timer >= 8.0:
                _advance_eq()

        # Explosion particles
        if sim.exploding:
            sim.expl_t += 1
            for part in sim.expl_parts:
                if part['vel'] is not None:
                    part['pos'] += part['vel']
                    part['vel'][1] -= 0.13
                part['life'] -= part['dr']
            if sim.expl_t > 150:
                sim.exploding = False
                sim.expl_parts.clear()

        # Camera orbit (same formula as HTML)
        ot      = sim.orbit_t
        cam_pos = np.array([math.cos(ot)*1080,
                             460 + math.sin(ot*0.35)*90,
                             math.sin(ot)*880])
        cam_rot = look_at(cam_pos, np.array([0., 150., 0.]))

        # Draw
        screen.fill(BG)
        draw_scene(screen, cam_pos, cam_rot)
        draw_hud(screen)
        if sim.eq_active:
            draw_eq_overlay(screen)
        draw_calc_flash(screen)
        pygame.display.flip()


if __name__ == '__main__':
    main()
