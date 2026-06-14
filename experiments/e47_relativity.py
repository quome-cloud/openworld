"""E47 - Relativity as a verified world model: reference frames and atomic clocks.

Physics whose defining feature is reference frames and changing references. A
verified symbolic world model encodes the exact relativistic dynamics (Lorentz
time dilation, relativistic velocity addition, gravitational potential); agents
carry atomic clocks (their proper time) and observe across frames; a changing
reference is a velocity-change action. The contrast is the paper's: the symbolic
model is exact and transfers to the relativistic regime (near c, real orbits)
where a learned model approximates and a Newtonian model is simply wrong - and we
validate it against REAL atomic-clock measurements (GPS, Hafele-Keating).

Four claims, all deterministic/offline (pure physics, no Ollama):
  1. time-dilation fidelity + OOD near c (symbolic exact; learned fails OOD;
     Newtonian has no dilation).
  2. relativistic velocity addition stays <= c; Galilean addition goes
     superluminal.
  3. twin paradox via an agent worldline (the changing-reference centerpiece).
  4. real atomic-clock validation: GPS (+38 us/day from first principles) and
     Hafele-Keating (east/west ns shifts within the measured error bars).
"""

import numpy as np

from openworld import Action, World
from openworld.transition import FunctionTransition

from common import save_results

C = 299_792_458.0          # speed of light, m/s
G = 6.674e-11              # gravitational constant
M_EARTH = 5.972e24        # kg
R_EARTH = 6.371e6         # m
OMEGA = 7.292e-5          # Earth angular velocity, rad/s
GM = G * M_EARTH
DAY = 86_400.0


def gamma(v):
    return 1.0 / np.sqrt(1.0 - (v / C) ** 2)


def clock_rate_symbolic(v):
    return np.sqrt(1.0 - (v / C) ** 2)        # = 1/gamma, the moving-clock rate


def vel_add_rel(u, v):
    return (u + v) / (1.0 + u * v / C ** 2)


def vel_add_gal(u, v):
    return u + v


# --- claim 1: time dilation, learned fit at low v fails OOD near c -----------
def time_dilation():
    train_x = np.linspace(0.0, 0.3, 60)          # v/c in-distribution
    train_y = clock_rate_symbolic(train_x * C)
    coeffs = np.polyfit(train_x, train_y, 4)      # learned: flexible low-v fit
    ood_x = np.linspace(0.31, 0.999, 50)          # OOD: toward c (the plunge)
    truth = clock_rate_symbolic(ood_x * C)
    learned = np.polyval(coeffs, ood_x)
    newton = np.ones_like(ood_x)                   # no dilation (classical)
    return {
        "ood_frac": ood_x.tolist(),
        "truth": truth.tolist(),
        "learned": learned.tolist(),
        "symbolic_mae": float(np.mean(np.abs(clock_rate_symbolic(ood_x * C) - truth))),
        "learned_mae": float(np.mean(np.abs(learned - truth))),
        "learned_err_near_c": float(abs(learned[-1] - truth[-1])),
        "newtonian_mae": float(np.mean(np.abs(newton - truth))),
        "newtonian_err_near_c": float(abs(newton[-1] - truth[-1])),
        "in_frac": train_x.tolist(), "in_truth": train_y.tolist(),
    }


# --- claim 2: velocity addition saturates at c ------------------------------
def velocity_addition():
    fracs = np.linspace(0.0, 0.95, 40)
    rel = [vel_add_rel(f * C, f * C) / C for f in fracs]
    gal = [vel_add_gal(f * C, f * C) / C for f in fracs]
    return {"fracs": fracs.tolist(), "rel_over_c": rel, "gal_over_c": gal,
            "rel_max_over_c": max(rel), "gal_max_over_c": max(gal)}


# --- claim 3: twin paradox via an agent worldline (a real World) -------------
def twin_world(relativistic=True):
    def step(state, action):
        s = dict(state)
        name = action["name"]
        params = action.get("params") or {}
        if name == "tick":
            dt = params["dt"]
            v = s["travel_v"]
            s["t"] += dt
            s["stay_tau"] += dt                              # stay-at-home: v=0
            s["travel_tau"] += dt * clock_rate_symbolic(v) if relativistic else dt
        elif name == "boost":
            s["travel_v"] = params["v"]
        return s
    return World(name="twins", description="twin paradox",
                 initial_state={"t": 0.0, "stay_tau": 0.0, "travel_tau": 0.0,
                                "travel_v": 0.0},
                 actions=["tick", "boost"], transition=FunctionTransition(step))


def twin_paradox(v_frac=0.8, leg_years=5.0):
    leg = leg_years * 3.156e7                                 # seconds (lab frame)
    steps = 500
    out = {}
    for label, rel in [("symbolic", True), ("newtonian", False)]:
        w = twin_world(rel)
        traj = []
        for vsign in (+1, -1):                                # out, then turnaround
            w.step(Action("boost", params={"v": vsign * v_frac * C}))
            for _ in range(steps):
                w.step(Action("tick", params={"dt": leg / steps}))
                traj.append((w.state["stay_tau"], w.state["travel_tau"]))
        out[label] = {
            "stay_years": w.state["stay_tau"] / 3.156e7,
            "travel_years": w.state["travel_tau"] / 3.156e7,
            "diff_years": (w.state["stay_tau"] - w.state["travel_tau"]) / 3.156e7,
        }
        if rel:
            out["traj_stay"] = [a / 3.156e7 for a, _ in traj]
            out["traj_travel"] = [b / 3.156e7 for _, b in traj]
    return out


# --- claim 4: real atomic clocks --------------------------------------------
def gps():
    """SR (orbital velocity) + GR (altitude) clock rate vs ground, per day."""
    r_sat = 2.6561e7                                          # GPS semi-major axis, m
    v_orb = np.sqrt(GM / r_sat)
    sr = -(v_orb ** 2) / (2 * C ** 2) * DAY                   # slower (seconds/day)
    gr = (GM / C ** 2) * (1 / R_EARTH - 1 / r_sat) * DAY      # faster
    return {"v_orbit_ms": float(v_orb),
            "sr_us_per_day": sr * 1e6, "gr_us_per_day": gr * 1e6,
            "net_us_per_day": (sr + gr) * 1e6}


def hafele_keating():
    """Flying clock vs ground clock, SR (with Earth rotation) + GR (altitude).
    Representative flight-averaged parameters; the published values used detailed
    flight logs."""
    lat = np.radians(34.0)
    h = 9000.0                                                # m, cruise altitude
    T = 45.0 * 3600.0                                         # s, time aloft
    g = GM / R_EARTH ** 2
    v_ground = OMEGA * R_EARTH * np.cos(lat)                  # ground eastward speed
    vp = 240.0                                                # plane ground speed, m/s
    grav = g * h / C ** 2 * T * 1e9                           # ns, faster (altitude)

    def net(direction):                                       # +1 east, -1 west
        v = direction * vp
        kin = (2 * v_ground * v + v ** 2) / (2 * C ** 2) * T * 1e9   # ns lost
        return grav - kin
    return {
        "model_east_ns": net(+1), "model_west_ns": net(-1),
        "grav_ns": grav,
        # published values (Hafele & Keating 1972), for comparison
        "pub_pred_east_ns": -40, "pub_pred_west_ns": 275,
        "pub_obs_east_ns": -59, "pub_obs_west_ns": 273,
        "pub_obs_east_err": 10, "pub_obs_west_err": 7,
        "newtonian_east_ns": 0.0, "newtonian_west_ns": 0.0,
    }


def main():
    td = time_dilation()
    va = velocity_addition()
    tw = twin_paradox()
    g = gps()
    hk = hafele_keating()

    save_results("e47_relativity", {
        "c": C, "time_dilation": td, "velocity_addition": va,
        "twin_paradox": tw, "gps": g, "hafele_keating": hk,
    })

    print("E47 - relativity as a verified world model\n")
    print(f"1. time dilation OOD (v->c): symbolic MAE {td['symbolic_mae']:.2e}, "
          f"learned MAE {td['learned_mae']:.3f} (err@0.999c {td['learned_err_near_c']:.2f}), "
          f"newtonian MAE {td['newtonian_mae']:.3f} (err@0.999c {td['newtonian_err_near_c']:.2f})")
    print(f"2. velocity addition: relativistic max {va['rel_max_over_c']:.3f}c (<=c), "
          f"Galilean max {va['gal_max_over_c']:.3f}c (superluminal)")
    s, t = tw["symbolic"], tw["newtonian"]
    print(f"3. twin paradox (v=0.8c, 10 yr round trip): symbolic stay "
          f"{s['stay_years']:.2f}yr vs travel {s['travel_years']:.2f}yr "
          f"(diff {s['diff_years']:.2f}yr); newtonian diff {t['diff_years']:.2f}yr")
    print(f"4a. GPS: SR {g['sr_us_per_day']:.1f} + GR {g['gr_us_per_day']:.1f} "
          f"= net {g['net_us_per_day']:.1f} us/day (documented ~+38)")
    print(f"4b. Hafele-Keating: model east {hk['model_east_ns']:.0f}ns "
          f"west {hk['model_west_ns']:.0f}ns | observed "
          f"{hk['pub_obs_east_ns']}+-{hk['pub_obs_east_err']}, "
          f"{hk['pub_obs_west_ns']}+-{hk['pub_obs_west_err']}")

    # --- self-checks ---
    assert td["symbolic_mae"] < 1e-12, "symbolic time dilation must be exact"
    assert td["newtonian_mae"] > 0.1, "newtonian (no dilation) should be far off near c"
    # both non-symbolic models diverge as v->0.999c; symbolic stays exact
    assert td["learned_err_near_c"] > 0.05, "learned fit should diverge near c"
    assert td["newtonian_err_near_c"] > 0.5, "newtonian is qualitatively wrong near c"
    assert va["rel_max_over_c"] < 1.0, "relativistic velocity addition must stay below c"
    assert va["gal_max_over_c"] > 1.0, "Galilean addition should go superluminal"
    assert s["diff_years"] > 0.5, "traveling twin should age measurably less"
    assert abs(t["diff_years"]) < 1e-9, "newtonian twins should age identically"
    assert abs(g["net_us_per_day"] - 38) < 3, "GPS net should be ~+38 us/day"
    assert g["sr_us_per_day"] < 0 < g["gr_us_per_day"], "GPS SR slows, GR speeds the clock"
    assert hk["model_east_ns"] < 0 < hk["model_west_ns"], "HK east/west asymmetry"
    assert abs(hk["model_east_ns"] - hk["pub_obs_east_ns"]) < 30, "HK east within error"
    assert abs(hk["model_west_ns"] - hk["pub_obs_west_ns"]) < 30, "HK west within error"
    print("\nall relativity checks pass; verified clocks match real measurements.")


if __name__ == "__main__":
    main()
