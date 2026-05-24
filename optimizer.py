# =========================================================
# optimizer.py
# Differential Evolution optimizer – maximises CL/CD ratio
# while keeping CL, CD, XCP within dataset-valid ranges.
# =========================================================

from scipy.optimize import differential_evolution
from predictor import aerodynamic_prediction

# =========================================================
# GLOBAL STATE
# =========================================================
convergence_history = []
generation_count    = 0


# =========================================================
# OBJECTIVE FUNCTION
# =========================================================

def objective_function(x, constraints=None):
    """
    Minimise  –(CL/CD)  subject to constraint penalties.
    Parameter order must match params_list in app.py exactly:
      [nose_len, body_len, wing_le, root_chord, tip_chord,
       semi_span, root_th, tip_th, wing_sweep,
       tail_le, root_chord1, tip_chord1, semi_span1,
       root_th1, tip_th1, mach, alpha, alt]
    """

    params = {
        'nose_len'    : x[0],
        'body_len'    : x[1],
        'wing_le'     : x[2],
        'root_chord'  : x[3],
        'tip_chord'   : x[4],
        'semi_span'   : x[5],
        'root_th'     : x[6],
        'tip_th'      : x[7],
        'wing_sweep'  : x[8],
        'tail_le'     : x[9],
        'root_chord1' : x[10],
        'tip_chord1'  : x[11],
        'semi_span1'  : x[12],
        'root_th1'    : x[13],
        'tip_th1'     : x[14],
        'mach'        : x[15],
        'alpha'       : x[16],
        'alt'         : x[17],
    }

    result = aerodynamic_prediction(params)

    cl  = result['CL']
    cd  = result['CD']
    xcp = result['XCP']

    # Avoid division by near-zero CD
    cd_safe = cd if abs(cd) > 1e-6 else 1e-6

    lift_to_drag = cl / cd_safe

    # ── Constraint penalty ─────────────────────────────────
    penalty = 0.0

    if constraints:

        cl_min, cl_max   = constraints.get('CL',  (-3.723,  15.2213))
        cd_min, cd_max   = constraints.get('CD',  (-1.187,   5.7352))
        xcp_min, xcp_max = constraints.get('XCP', (-12.3114, -3.5322))

        if not (cl_min  <= cl  <= cl_max):
            penalty += 1000.0 * max(cl_min - cl, cl - cl_max, 0)

        if not (cd_min  <= cd  <= cd_max):
            penalty += 1000.0 * max(cd_min - cd, cd - cd_max, 0)

        if not (xcp_min <= xcp <= xcp_max):
            penalty += 1000.0 * max(xcp_min - xcp, xcp - xcp_max, 0)

    return -(lift_to_drag) + penalty


# =========================================================
# CALLBACK
# =========================================================

def _store_convergence(xk, convergence):
    global generation_count
    generation_count += 1
    convergence_history.append({
        'generation': generation_count,
        'fitness'   : float(convergence),
    })


# =========================================================
# MAIN OPTIMISATION FUNCTION
# =========================================================

def run_optimization(
    bounds,
    maxiter=50,
    popsize=15,
    constraints=None
):
    """
    Run Differential Evolution to find geometry + flight
    conditions that maximise CL/CD within the given bounds
    and output constraints.

    Parameters
    ----------
    bounds      : list of (low, high) tuples, one per parameter
    maxiter     : int   – maximum number of generations
    popsize     : int   – population multiplier
    constraints : dict  – {'CL': (min,max), 'CD': (min,max), 'XCP': (min,max)}

    Returns
    -------
    result               : scipy OptimizeResult
    convergence_history  : list of {'generation': int, 'fitness': float}
    """

    global convergence_history, generation_count
    convergence_history = []
    generation_count    = 0

    result = differential_evolution(
        func       = lambda x: objective_function(x, constraints),
        bounds     = bounds,
        maxiter    = maxiter,
        popsize    = popsize,
        polish     = True,
        disp       = True,
        updating   = 'deferred',
        seed       = 42,
        callback   = _store_convergence,
        tol        = 1e-6,
        mutation   = (0.5, 1.0),
        recombination = 0.7,
    )

    return result, convergence_history
