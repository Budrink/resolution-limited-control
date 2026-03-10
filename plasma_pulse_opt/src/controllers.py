"""
Piecewise-constant controllers: constant, PI, pulse train, kick-and-hold, event-triggered.
All return P in [0, P_max]. Callables take (t, W, B) and optional state.
"""

import numpy as np
from typing import Callable, Tuple, Any


def clip_power(P: float, P_max: float) -> float:
    return float(np.clip(P, 0.0, P_max))


# ---- A) Constant ----
def make_constant(P_const: float, P_max: float) -> Callable:
    """P(t) = P_const."""
    P = clip_power(P_const, P_max)
    def controller(t: float, W: float, B: float, state: Any = None) -> Tuple[float, Any]:
        return P, state
    return controller


# ---- B) PI baseline ----
def make_pi(W_target: float, Kp: float, Ki: float, P_base: float, P_max: float) -> Callable:
    """P = clip(P_base + Kp*error + Ki*integral(error), 0, P_max). State = (integral,)."""
    def controller(t: float, W: float, B: float, state: Any = None) -> Tuple[float, Any]:
        if state is None:
            integral = 0.0
        else:
            integral = state[0]
        error = W_target - W
        integral = integral + error  # will be scaled by dt in the caller if needed
        P = P_base + Kp * error + Ki * integral
        P = clip_power(P, P_max)
        return P, (integral,)
    return controller


def pi_integral_update(integral: float, error: float, dt: float) -> float:
    """Euler update for integral of error."""
    return integral + error * dt


# PI that manages its own integral with dt (called each step)
class PIController:
    def __init__(self, W_target: float, Kp: float, Ki: float, P_base: float, P_max: float):
        self.W_target = W_target
        self.Kp = Kp
        self.Ki = Ki
        self.P_base = P_base
        self.P_max = P_max
        self.integral = 0.0

    def __call__(self, t: float, W: float, B: float, dt: float) -> float:
        error = self.W_target - W
        self.integral = self.integral + error * dt
        P = self.P_base + self.Kp * error + self.Ki * self.integral
        return clip_power(P, self.P_max)


# ---- C) Rectangular pulse train ----
def make_pulse_train(P_base: float, DeltaP: float, period: float, duty: float,
                     P_max: float) -> Callable:
    """P = P_base + DeltaP during on-window, else P_base. duty in (0,1)."""
    P_lo = clip_power(P_base, P_max)
    P_hi = clip_power(P_base + DeltaP, P_max)
    ton = period * duty

    def controller(t: float, W: float, B: float, state: Any = None) -> Tuple[float, Any]:
        phase = (t % period) if period > 0 else 0.0
        P = P_hi if phase < ton else P_lo
        return P, state
    return controller


# ---- D) Kick-and-hold ----
def make_kick_hold(P_kick: float, t_kick: float, P_hold: float, period: float,
                   P_max: float) -> Callable:
    """Each period: P = P_kick for t_kick, then P_hold until next period."""
    P_k = clip_power(P_kick, P_max)
    P_h = clip_power(P_hold, P_max)

    def controller(t: float, W: float, B: float, state: Any = None) -> Tuple[float, Any]:
        phase = t % period if period > 0 else 0.0
        P = P_k if phase < t_kick else P_h
        return P, state
    return controller


# ---- E) Event-triggered recharge ----
def make_event_recharge(P_hold: float, P_kick: float, t_kick: float,
                        B_min: float, refractory: float, P_max: float) -> Callable:
    """If B < B_min and refractory passed, apply kick for t_kick then hold."""
    P_h = clip_power(P_hold, P_max)
    P_k = clip_power(P_kick, P_max)
    # state: (t_last_kick_end, in_kick_phase, kick_until)
    def controller(t: float, W: float, B: float, state: Any = None) -> Tuple[float, Any]:
        if state is None:
            t_last_kick_end = -1e9
            in_kick = False
            kick_until = -1e9
        else:
            t_last_kick_end, in_kick, kick_until = state
        if in_kick and t < kick_until:
            return P_k, (t_last_kick_end, True, kick_until)
        if in_kick and t >= kick_until:
            in_kick = False
            t_last_kick_end = t
        if not in_kick and B < B_min and (t - t_last_kick_end) >= refractory:
            in_kick = True
            kick_until = t + t_kick
            return P_k, (t_last_kick_end, True, kick_until)
        return P_h, (t_last_kick_end, in_kick, kick_until)
    return controller


def get_controller_P(controller: Callable, t: float, W: float, B: float,
                     state: Any, dt: float) -> Tuple[float, Any]:
    """
    Call controller and return (P, new_state).
    If controller is PIController, use dt for integral; else state is passed through.
    """
    if isinstance(controller, PIController):
        P = controller(t, W, B, dt)
        return P, state
    return controller(t, W, B, state)
