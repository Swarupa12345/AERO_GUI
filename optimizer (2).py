# =========================================================
# optimizer.py  —  Differential Evolution  (updated per
#                  de_pipeline_output_description.docx)
#
# KEY CHANGES from previous version:
#   • maxiter = 50 (was 8), popsize = 10 (was 5)
#   • Log format matches docx: "Gen 01 – best fitness: X.XXXXXX"
#   • Convergence callback emits docx-style fitness lines
#   • Timing added around differential_evolution() call
#   • Output artefacts: best_geometry.csv, full_performance.csv,
#     summary_metrics.csv, evolution_history.xlsx saved to out_dir
#   • run_optimization() accepts out_dir parameter
# =========================================================

import os
import time
import csv

import numpy as np
from scipy.optimize import differential_evolution
from predictor import aerodynamic_prediction

# =========================================================
# GLOBAL STATE  (reset each run)
# =========================================================
convergence_history = []   # [{'generation': int, 'fitness': float}]
generation_count    = 0
_log_callback       = None   # optional callable(msg: str)


# =========================================================
# OBJECTIVE FUNCTION
# =========================================================
def objective_function(x, constraints=None):
    """
    Minimise −(CL/CD) subject to constraint penalties.
    Parameter order:
      [nose_len, body_len, wing_le, root_chord, tip_chord,
       semi_span, root_th, tip_th, wing_sweep,
       tail_le, root_chord1, tip_chord1, semi_span1,
       root_th1, tip_th1, mach, alpha, alt]
    """
    params = {
        'nose_len'   : x[0],  'body_len'   : x[1],
        'wing_le'    : x[2],  'root_chord' : x[3],
        'tip_chord'  : x[4],  'semi_span'  : x[5],
        'root_th'    : x[6],  'tip_th'     : x[7],
        'wing_sweep' : x[8],  'tail_le'    : x[9],
        'root_chord1': x[10], 'tip_chord1' : x[11],
        'semi_span1' : x[12], 'root_th1'   : x[13],
        'tip_th1'    : x[14], 'mach'       : x[15],
        'alpha'      : x[16], 'alt'        : x[17],
    }

    result  = aerodynamic_prediction(params)
    cl, cd, xcp = result['CL'], result['CD'], result['XCP']
    cd_safe = cd if abs(cd) > 1e-6 else 1e-6
    ld      = cl / cd_safe

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

    return -ld + penalty


# =========================================================
# CONVERGENCE CALLBACK
# =========================================================
def _store_convergence(xk, convergence):
    """Called by scipy after every generation."""
    global generation_count
    generation_count += 1
    best_ld = -objective_function(xk)   # recover +CL/CD
    convergence_history.append({
        'generation': generation_count,
        'fitness'   : round(float(best_ld), 6),
    })
    # Emit docx-style log line
    line = (f'Gen {generation_count:02d} – '
            f'best fitness: {best_ld:.6f}')
    if _log_callback:
        _log_callback(line)


# =========================================================
# MAIN OPTIMISATION FUNCTION
# =========================================================
def run_optimization(
    bounds,
    maxiter    = 50,        # ← docx default: 50 generations
    popsize    = 10,        # ← docx default: popsize 10
    constraints= None,
    out_dir    = None,      # if set, saves output artefacts
    log_callback = None,    # callable(str) for live logging
):
    """
    Run Differential Evolution to maximise CL/CD.

    Returns
    -------
    result              : scipy OptimizeResult
    convergence_history : list of {'generation', 'fitness'}
    elapsed             : float  (seconds)
    """
    global convergence_history, generation_count, _log_callback

    convergence_history = []
    generation_count    = 0
    _log_callback       = log_callback

    t_start = time.perf_counter()

    result = differential_evolution(
        func          = lambda x: objective_function(x, constraints),
        bounds        = bounds,
        maxiter       = maxiter,
        popsize       = popsize,
        polish        = True,
        disp          = True,      # prints scipy's own step lines to stdout
        updating      = 'deferred',
        seed          = 42,
        callback      = _store_convergence,
        tol           = 1e-6,
        mutation      = (0.5, 1.0),
        recombination = 0.7,
    )

    elapsed = round(time.perf_counter() - t_start, 4)

    # ── Save output artefacts (matches docx section 3) ────
    if out_dir:
        _save_artefacts(result, convergence_history, out_dir, elapsed)

    return result, convergence_history, elapsed


# =========================================================
# ARTEFACT SAVER
# =========================================================
PARAM_NAMES = [
    'nose_len','body_len','wing_le','root_chord','tip_chord',
    'semi_span','root_th','tip_th','wing_sweep',
    'tail_le','root_chord1','tip_chord1','semi_span1',
    'root_th1','tip_th1','mach','alpha','alt',
]

def _save_artefacts(result, history, out_dir, elapsed):
    os.makedirs(out_dir, exist_ok=True)

    best_x   = result.x
    best_ld  = -result.fun
    best_prm = {n: round(float(v), 6)
                for n, v in zip(PARAM_NAMES, best_x)}
    best_res = aerodynamic_prediction(best_prm)

    # 1. best_geometry.csv
    with open(os.path.join(out_dir,'best_geometry.csv'),
              'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['parameter','value'])
        for k, v in best_prm.items():
            w.writerow([k, v])

    # 2. full_performance.csv
    with open(os.path.join(out_dir,'full_performance.csv'),
              'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['CL','CD','XCP','CL_CD','elapsed_s'])
        w.writerow([best_res['CL'], best_res['CD'],
                    best_res['XCP'], round(best_ld,6),
                    elapsed])

    # 3. summary_metrics.csv
    with open(os.path.join(out_dir,'summary_metrics.csv'),
              'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric','value'])
        w.writerow(['best_CL_CD', round(best_ld,6)])
        w.writerow(['generations', len(history)])
        w.writerow(['elapsed_s',  elapsed])
        w.writerow(['optimizer_success', result.success])

    # 4. evolution_history.xlsx  (via openpyxl if available)
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = 'Evolution'
        ws.append(['generation','fitness'])
        for h in history:
            ws.append([h['generation'], h['fitness']])
        wb.save(os.path.join(out_dir,'evolution_history.xlsx'))
    except ImportError:
        # fallback: plain CSV
        with open(os.path.join(out_dir,'evolution_history.csv'),
                  'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['generation','fitness'])
            for h in history:
                w.writerow([h['generation'], h['fitness']])
