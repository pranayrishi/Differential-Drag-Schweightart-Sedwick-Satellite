"""
Validation tests for the Traub corrected SS solution.

Five tests:
  1. CW reduction (c=1): Traub solution matches numerical reference to
     near machine precision when omega_SS = n (i.e. c = 1 exactly).
  2. No-force agreement: both Shao and Traub match numerical (all 4 cases
     where fy=0 make Shao identical to Traub).
  3. Radial force only: both Shao and Traub match numerical (fy=0).
  4. Tangential force, c=1: Shao == Traub (exact, since omega_SS = n).
  5. Error scaling: Shao delta_y error at t=5T scales linearly with |c-1|
     (confirmed via log-log fit with slope ≈ 1).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dynamics.constants import mu, R_E, J2
from dynamics.ss_model import compute_mean_motion, compute_SS_coefficient
from dynamics.numerical_integrator import SS_numerical_integration
from dynamics.analytical_traub import SS_analytical_traub
from dynamics.analytical_shao import SS_analytical_shao


# ------------------------------------------------------------------ #
# Helper                                                               #
# ------------------------------------------------------------------ #

def _run_case(t_eval, y0, delta_fx, delta_fy, n, c):
    """Return (sol_num_y, dx_traub, dy_traub, dx_shao, dy_shao)."""
    dx0, dxd0, dy0, dyd0 = y0[0], y0[1], y0[2], y0[3]

    _, sol_num = SS_numerical_integration(
        (t_eval[0], t_eval[-1]), y0, delta_fx, delta_fy, 0.0,
        n, c, t_eval=t_eval)

    dx_t, _, dy_t, _ = SS_analytical_traub(
        t_eval, dx0, dxd0, dy0, dyd0, delta_fx, delta_fy, n, c)

    dx_s, _, dy_s, _ = SS_analytical_shao(
        t_eval, dx0, dxd0, dy0, dyd0, delta_fx, delta_fy, n, c)

    return sol_num, dx_t, dy_t, dx_s, dy_s


def _print_result(name, passed, detail=''):
    status = 'PASS' if passed else 'FAIL'
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")


# ------------------------------------------------------------------ #
# Test 1: CW reduction when c = 1                                      #
# ------------------------------------------------------------------ #

def test_CW_reduction(n, T):
    """
    With c=1 (omega_SS = n), the Traub solution is the exact closed-form
    solution of the HCW equations.  Compare against high-accuracy numerical
    integration rather than a textbook formula (textbook y-particular
    solutions vary by sign convention).
    """
    c = 1.0
    t_eval = np.linspace(0, 5 * T, 5000)

    delta_x0    = 150.0
    delta_xdot0 = 0.05
    delta_y0    = -200.0
    delta_ydot0 = 0.1
    delta_fx    = 3e-7
    delta_fy    = 2e-7

    y0 = [delta_x0, delta_xdot0, delta_y0, delta_ydot0, 0.0, 0.0]

    _, sol_num = SS_numerical_integration(
        (t_eval[0], t_eval[-1]), y0, delta_fx, delta_fy, 0.0,
        n, c, t_eval=t_eval)

    dx_t, _, dy_t, _ = SS_analytical_traub(
        t_eval, delta_x0, delta_xdot0, delta_y0, delta_ydot0,
        delta_fx, delta_fy, n, c)

    err_x = np.max(np.abs(dx_t - sol_num[0]))
    err_y = np.max(np.abs(dy_t - sol_num[2]))

    # Also verify Traub == Shao when c=1 (they should be identical)
    dx_s, _, dy_s, _ = SS_analytical_shao(
        t_eval, delta_x0, delta_xdot0, delta_y0, delta_ydot0,
        delta_fx, delta_fy, n, c)
    diff_ts_x = np.max(np.abs(dx_t - dx_s))
    diff_ts_y = np.max(np.abs(dy_t - dy_s))

    tol_num  = 1e-6   # m — against numerical (DOP853 accuracy)
    tol_ts   = 1e-10  # m — between Traub and Shao (should be identical)

    passed = (err_x < tol_num and err_y < tol_num
              and diff_ts_x < tol_ts and diff_ts_y < tol_ts)
    _print_result(
        'CW reduction (c=1): Traub matches numerical and equals Shao',
        passed,
        (f'Traub vs num: max |dx|={err_x:.2e} m, |dy|={err_y:.2e} m '
         f'[tol={tol_num:.0e}]  |  '
         f'Traub vs Shao: |dx|={diff_ts_x:.2e}, |dy|={diff_ts_y:.2e} '
         f'[tol={tol_ts:.0e}]'))
    return passed


# ------------------------------------------------------------------ #
# Test 2: No-force agreement                                           #
# ------------------------------------------------------------------ #

def test_no_force(n, c, T, y0, t_eval):
    """Both Shao and Traub match numerical when forces are zero."""
    sol_num, dx_t, dy_t, dx_s, dy_s = _run_case(
        t_eval, y0, 0.0, 0.0, n, c)

    err_t_x = np.max(np.abs(dx_t - sol_num[0]))
    err_t_y = np.max(np.abs(dy_t - sol_num[2]))
    err_s_x = np.max(np.abs(dx_s - sol_num[0]))
    err_s_y = np.max(np.abs(dy_s - sol_num[2]))

    tol = 1e-6
    passed = all(e < tol for e in [err_t_x, err_t_y, err_s_x, err_s_y])
    _print_result(
        'No-force: Traub and Shao agree with numerical',
        passed,
        (f'Traub max err: dx={err_t_x:.2e}, dy={err_t_y:.2e}  |  '
         f'Shao max err: dx={err_s_x:.2e}, dy={err_s_y:.2e}  [tol={tol:.0e}]'))
    return passed


# ------------------------------------------------------------------ #
# Test 3: Radial force only                                            #
# ------------------------------------------------------------------ #

def test_radial_force(n, c, T, y0, t_eval):
    """Both Shao and Traub match numerical when only radial force is applied."""
    sol_num, dx_t, dy_t, dx_s, dy_s = _run_case(
        t_eval, y0, 1e-7, 0.0, n, c)

    err_t_x = np.max(np.abs(dx_t - sol_num[0]))
    err_t_y = np.max(np.abs(dy_t - sol_num[2]))
    err_s_x = np.max(np.abs(dx_s - sol_num[0]))
    err_s_y = np.max(np.abs(dy_s - sol_num[2]))

    tol = 1e-6
    passed = all(e < tol for e in [err_t_x, err_t_y, err_s_x, err_s_y])
    _print_result(
        'Radial force only: Traub and Shao agree with numerical',
        passed,
        (f'Traub max err: dx={err_t_x:.2e}, dy={err_t_y:.2e}  |  '
         f'Shao max err: dx={err_s_x:.2e}, dy={err_s_y:.2e}  [tol={tol:.0e}]'))
    return passed


# ------------------------------------------------------------------ #
# Test 4: Tangential force, c = 1 (both solutions identical)          #
# ------------------------------------------------------------------ #

def test_tangential_c1(n, T):
    """When c=1, omega_SS=n and B_Traub = B_Shao, so solutions must be identical."""
    c = 1.0
    delta_x0    = 100.0
    delta_xdot0 = 0.0
    delta_y0    = 0.0
    delta_ydot0 = -2.0 * c * n * delta_x0
    delta_fx    = 0.0
    delta_fy    = 1e-7

    t_eval = np.linspace(0, 5 * T, 5000)

    dx_t, _, dy_t, _ = SS_analytical_traub(
        t_eval, delta_x0, delta_xdot0, delta_y0, delta_ydot0,
        delta_fx, delta_fy, n, c)
    dx_s, _, dy_s, _ = SS_analytical_shao(
        t_eval, delta_x0, delta_xdot0, delta_y0, delta_ydot0,
        delta_fx, delta_fy, n, c)

    diff_x = np.max(np.abs(dx_t - dx_s))
    diff_y = np.max(np.abs(dy_t - dy_s))

    tol = 1e-10
    passed = (diff_x < tol) and (diff_y < tol)
    _print_result(
        'Tangential force, c=1: Traub == Shao',
        passed,
        f'max |dx diff| = {diff_x:.2e}, max |dy diff| = {diff_y:.2e}  [tol={tol:.0e}]')
    return passed


# ------------------------------------------------------------------ #
# Test 5: Error scaling — Shao error ∝ (c - 1)                       #
# ------------------------------------------------------------------ #

def test_error_scaling(a, e, n, T):
    """
    The Shao delta_y error at t=5T for Case C should scale linearly with |c-1|.
    Fit log-log slope; accept if slope is between 0.7 and 1.3.
    """
    inclinations_deg = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 110.0, 130.0])

    delta_x0    = 100.0
    delta_xdot0 = 0.0
    delta_y0    = 0.0
    delta_fx    = 0.0
    delta_fy    = 1e-7

    t_eval = np.linspace(0, 5 * T, 2000)

    cm1_list = []
    err_list = []

    for i_deg in inclinations_deg:
        i_rad = np.radians(i_deg)
        c = compute_SS_coefficient(a, e, i_rad)

        omega_SS_sq = n**2 * (4.0 * c + 2.0 - 5.0 * c**2)
        if omega_SS_sq <= 0 or np.abs(c - 1.0) < 1e-9:
            continue

        dyd0 = -2.0 * c * n * delta_x0
        y0   = [delta_x0, delta_xdot0, delta_y0, dyd0, 0.0, 0.0]

        _, sol_num = SS_numerical_integration(
            (t_eval[0], t_eval[-1]), y0, delta_fx, delta_fy, 0.0,
            n, c, t_eval=t_eval)

        _, _, dy_s, _ = SS_analytical_shao(
            t_eval, delta_x0, delta_xdot0, delta_y0, dyd0,
            delta_fx, delta_fy, n, c)

        err_y = np.abs(dy_s[-1] - sol_num[2, -1])
        if err_y > 1e-12:
            cm1_list.append(np.abs(c - 1.0))
            err_list.append(err_y)

    if len(cm1_list) < 3:
        _print_result('Error scaling: Shao error ~ |c-1|', False,
                      'Insufficient data points')
        return False

    log_cm1 = np.log10(np.array(cm1_list))
    log_err = np.log10(np.array(err_list))
    coeffs  = np.polyfit(log_cm1, log_err, 1)
    slope   = coeffs[0]

    passed = (0.7 <= slope <= 1.3)
    _print_result(
        'Error scaling: Shao delta_y error ~ |c-1|',
        passed,
        f'Log-log slope = {slope:.3f}  (expected ~1.0, accepted range [0.7, 1.3])')
    return passed


# ------------------------------------------------------------------ #
# Master runner                                                        #
# ------------------------------------------------------------------ #

def run_all_tests(n, c, t_eval, y0, a=6871.0e3, e=0.0):
    """
    Run all five validation tests and print a summary.

    Parameters
    ----------
    n : float
        Mean motion [rad/s]
    c : float
        SS coefficient for nominal orbit
    t_eval : ndarray
        Time array [s]
    y0 : list
        Nominal initial state [dx, dxd, dy, dyd, dz, dzd]
    a : float, optional
        Semi-major axis [m]
    e : float, optional
        Eccentricity [-]

    Returns
    -------
    results : list of bool
        Pass/fail for each of the five tests.
    """
    T = 2.0 * np.pi / n

    print('\n' + '=' * 60)
    print('  VALIDATION TESTS — Traub et al. (2025) SS Solution')
    print('=' * 60)

    results = []
    results.append(test_CW_reduction(n, T))
    results.append(test_no_force(n, c, T, y0, t_eval))
    results.append(test_radial_force(n, c, T, y0, t_eval))
    results.append(test_tangential_c1(n, T))
    results.append(test_error_scaling(a, e, n, T))

    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print('=' * 60)
    print(f'  Results: {n_pass}/{len(results)} PASSED, {n_fail} FAILED')
    print('=' * 60 + '\n')

    return results


# ------------------------------------------------------------------ #
# Standalone execution                                                 #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    a = 6871.0e3
    e = 0.0
    i = np.radians(98.0)
    n = compute_mean_motion(a)
    T = 2.0 * np.pi / n
    c = compute_SS_coefficient(a, e, i)

    delta_x0    = 100.0
    delta_xdot0 = 0.0
    delta_y0    = 0.0
    delta_ydot0 = -2.0 * c * n * delta_x0

    y0     = [delta_x0, delta_xdot0, delta_y0, delta_ydot0, 0.0, 0.0]
    t_eval = np.linspace(0, 5 * T, 5000)

    run_all_tests(n, c, t_eval, y0, a=a, e=e)
