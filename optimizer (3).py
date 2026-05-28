# =========================================================
# optimizer.py  v4.0
# Differential Evolution optimizer — maximises CL/CD ratio
# while keeping CL, CD, XCP within dataset-valid ranges.
#
# FIXES & ENHANCEMENTS over v3.0:
#   BUG FIX : maxiter and popsize were HARDCODED as 8 and 5
#             inside differential_evolution() — now uses the
#             values passed by the caller (GUI / CLI).
#   NEW     : convergence_history now also stores best CL,
#             CD, XCP per generation → enables aero-metrics
#             convergence plot in the GUI.
#   NEW     : _all_evaluations list tracks every objective-
#             function call → enables performance scatter plot.
#   NEW     : export_results() saves artefacts exactly as
#             described in de_pipeline_output_description.docx:
#               best_geometry.csv, full_performance.csv,
#               summary_metrics.csv, evolution_history.xlsx
# =========================================================

import os
import pandas as pd
from scipy.optimize import differential_evolution
from predictor import aerodynamic_prediction

# =========================================================
# PARAMETER ORDER  (must match app.py's params_list exactly)
# =========================================================
PARAMS_LIST = [
    'nose_len', 'body_len', 'wing_le', 'root_chord', 'tip_chord',
    'semi_span', 'root_th', 'tip_th', 'wing_sweep',
    'tail_le', 'root_chord1', 'tip_chord1', 'semi_span1',
    'root_th1', 'tip_th1', 'mach', 'alpha', 'alt',
]

LABELS = {
    'nose_len'    : 'nose length',
    'body_len'    : 'body_length',
    'wing_le'     : 'wing LE',
    'root_chord'  : 'root chord',
    'tip_chord'   : 'tip chord',
    'semi_span'   : 'semi-span',
    'root_th'     : 'root th',
    'tip_th'      : 'tip th',
    'wing_sweep'  : 'wing sweep',
    'tail_le'     : 'tail LE',
    'root_chord1' : 'root chord.1',
    'tip_chord1'  : 'tip chord.1',
    'semi_span1'  : 'semi-span.1',
    'root_th1'    : 'root th.1',
    'tip_th1'     : 'tip th.1',
    'mach'        : 'MACH',
    'alpha'       : 'ALPHA',
    'alt'         : 'ALT',
}

# =========================================================
# GLOBAL STATE — reset at the start of every optimisation
# =========================================================
convergence_history: list = []   # list of per-generation dicts
_all_evaluations:   list = []    # list of every obj-fn evaluation
_generation_count:  int  = 0


# =========================================================
# OBJECTIVE FUNCTION
# =========================================================
def objective_function(x, constraints=None):
    """
    Minimise  –(CL/CD)  subject to constraint penalties.

    Parameter order matches PARAMS_LIST above.

    Returns the scalar objective value (negative CL/CD + penalty).
    Side effect: appends to _all_evaluations for scatter plot.
    """
    params = {p: x[i] for i, p in enumerate(PARAMS_LIST)}

    result = aerodynamic_prediction(params)
    cl  = result['CL']
    cd  = result['CD']
    xcp = result['XCP']

    cd_safe     = cd if abs(cd) > 1e-6 else 1e-6
    lift_to_drag = cl / cd_safe

    # ── Constraint penalty ────────────────────────────────
    penalty = 0.0
    if constraints:
        cl_min,  cl_max  = constraints.get('CL',  (-3.723,  15.2213))
        cd_min,  cd_max  = constraints.get('CD',  (-1.187,   5.7352))
        xcp_min, xcp_max = constraints.get('XCP', (-12.3114, -3.5322))

        if not (cl_min  <= cl  <= cl_max):
            penalty += 1000.0 * max(cl_min - cl,  cl  - cl_max,  0)
        if not (cd_min  <= cd  <= cd_max):
            penalty += 1000.0 * max(cd_min - cd,  cd  - cd_max,  0)
        if not (xcp_min <= xcp <= xcp_max):
            penalty += 1000.0 * max(xcp_min - xcp, xcp - xcp_max, 0)

    obj_val = -(lift_to_drag) + penalty

    # Track every evaluation for the scatter plot
    _all_evaluations.append({
        'CL'      : cl,
        'CD'      : cd,
        'XCP'     : xcp,
        'CLCD'    : lift_to_drag,
        'penalty' : penalty,
    })

    return obj_val


# =========================================================
# CONVERGENCE CALLBACK
# Called once per generation by differential_evolution.
# xk = current best solution vector
# convergence = fractional spread of the population (scipy)
# =========================================================
def _store_convergence(xk, convergence):
    """
    Record per-generation best solution.

    Evaluates xk through aerodynamic_prediction() to get the
    physical CL / CD / XCP at the best point found so far.
    Fitness is reported as positive CL/CD (matches docx format).
    """
    global _generation_count
    _generation_count += 1

    params = {p: float(xk[i]) for i, p in enumerate(PARAMS_LIST)}
    pred   = aerodynamic_prediction(params)

    cl   = pred['CL']
    cd   = pred['CD']
    xcp  = pred['XCP']
    cd_s = cd if abs(cd) > 1e-6 else 1e-6

    convergence_history.append({
        'generation': _generation_count,
        'fitness'   : round(cl / cd_s, 6),   # positive CL/CD — matches docx
        'CL'        : cl,
        'CD'        : cd,
        'XCP'       : xcp,
    })


# =========================================================
# MAIN OPTIMISATION FUNCTION
# =========================================================
def run_optimization(
    bounds,
    maxiter=50,
    popsize=15,
    constraints=None,
):
    """
    Run Differential Evolution to find geometry + flight
    conditions that maximise CL/CD within the given bounds
    and output constraints.

    Parameters
    ----------
    bounds      : list of (low, high) per parameter (18 entries)
    maxiter     : int   – maximum generations
    popsize     : int   – population multiplier (total pop = popsize × 18)
    constraints : dict  – {'CL': (min,max), 'CD': (min,max), 'XCP': (min,max)}

    Returns
    -------
    result              : scipy OptimizeResult
    convergence_history : list of per-generation dicts
                          keys: generation, fitness, CL, CD, XCP
    """
    global convergence_history, _all_evaluations, _generation_count

    # Reset global state
    convergence_history = []
    _all_evaluations    = []
    _generation_count   = 0

    result = differential_evolution(
        func          = lambda x: objective_function(x, constraints),
        bounds        = bounds,
        maxiter       = maxiter,       # ← BUG FIX: was hardcoded as 8
        popsize       = popsize,       # ← BUG FIX: was hardcoded as 5
        polish        = True,
        disp          = True,
        updating      = 'deferred',
        seed          = 42,
        callback      = _store_convergence,
        tol           = 1e-6,
        mutation      = (0.5, 1.0),
        recombination = 0.7,
    )

    return result, convergence_history


# =========================================================
# EXPORT HELPERS  (matches docx artefact list exactly)
# =========================================================
def export_results(result, convergence_hist, output_dir='.'):
    """
    Save optimisation artefacts to output_dir:
      1. best_geometry.csv        — optimal 18-parameter row
      2. full_performance.csv     — CL, CD, XCP at optimal point
      3. summary_metrics.csv      — scalar summary
      4. evolution_history.xlsx   — full per-generation history

    Parameters
    ----------
    result          : scipy OptimizeResult
    convergence_hist: list returned by run_optimization()
    output_dir      : folder to write files into (created if absent)

    Returns
    -------
    dict of written file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    # ── 1. best_geometry.csv ─────────────────────────────
    best_params = {LABELS[p]: round(float(result.x[i]), 6)
                   for i, p in enumerate(PARAMS_LIST)}
    pd.DataFrame([best_params]).to_csv(
        os.path.join(output_dir, 'best_geometry.csv'), index=False)
    paths['best_geometry'] = os.path.join(output_dir, 'best_geometry.csv')

    # ── 2. full_performance.csv ──────────────────────────
    best_prm  = {p: float(result.x[i]) for i, p in enumerate(PARAMS_LIST)}
    best_pred = aerodynamic_prediction(best_prm)
    perf = {
        'CL'          : best_pred['CL'],
        'CD'          : best_pred['CD'],
        'XCP'         : best_pred['XCP'],
        'CL_CD_ratio' : round(best_pred['CL'] / max(abs(best_pred['CD']), 1e-6), 6),
        'source'      : best_pred.get('source', '—'),
    }
    pd.DataFrame([perf]).to_csv(
        os.path.join(output_dir, 'full_performance.csv'), index=False)
    paths['full_performance'] = os.path.join(output_dir, 'full_performance.csv')

    # ── 3. summary_metrics.csv ───────────────────────────
    summary = {
        'best_CL_CD_ratio'   : round(-result.fun, 6),
        'n_generations'      : len(convergence_hist),
        'n_evaluations'      : len(_all_evaluations),
        'converged'          : bool(result.success),
        'best_CL'            : best_pred['CL'],
        'best_CD'            : best_pred['CD'],
        'best_XCP'           : best_pred['XCP'],
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, 'summary_metrics.csv'), index=False)
    paths['summary_metrics'] = os.path.join(output_dir, 'summary_metrics.csv')

    # ── 4. evolution_history.xlsx ────────────────────────
    hist_df = pd.DataFrame(convergence_hist)
    hist_df.to_excel(
        os.path.join(output_dir, 'evolution_history.xlsx'), index=False)
    paths['evolution_history'] = os.path.join(output_dir, 'evolution_history.xlsx')

    return paths
