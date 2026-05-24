# =========================================================
# envelope.py
# Flight envelope sweep functions.
# Each sweep varies one parameter while holding the rest
# fixed at the values supplied by the GUI (base_params).
# =========================================================

import numpy as np
from predictor import aerodynamic_prediction


# =========================================================
# ALPHA SWEEP
# =========================================================

def alpha_sweep(base_params, alpha_min, alpha_max, step):
    """
    Vary angle-of-attack from alpha_min to alpha_max in
    increments of step, holding all other params fixed.

    Returns a list of dicts:
        {'alpha': float, 'CL': float, 'CD': float, 'XCP': float}
    """

    results = []

    step = step if step > 0 else 1.0

    for alpha in np.arange(alpha_min, alpha_max + step * 0.5, step):

        params = base_params.copy()
        params['alpha'] = round(float(alpha), 4)

        pred = aerodynamic_prediction(params)

        results.append({
            'alpha' : round(float(alpha), 4),
            'CL'    : pred['CL'],
            'CD'    : pred['CD'],
            'XCP'   : pred['XCP'],
        })

    return results


# =========================================================
# MACH SWEEP
# =========================================================

def mach_sweep(base_params, mach_min, mach_max, step):
    """
    Vary Mach number from mach_min to mach_max in increments
    of step, holding all other params fixed.

    Returns a list of dicts:
        {'mach': float, 'CL': float, 'CD': float, 'XCP': float}
    """

    results = []

    step = step if step > 0 else 0.1

    for mach in np.arange(mach_min, mach_max + step * 0.5, step):

        params = base_params.copy()
        params['mach'] = round(float(mach), 4)

        pred = aerodynamic_prediction(params)

        results.append({
            'mach': round(float(mach), 4),
            'CL'  : pred['CL'],
            'CD'  : pred['CD'],
            'XCP' : pred['XCP'],
        })

    return results


# =========================================================
# ALTITUDE SWEEP
# =========================================================

def altitude_sweep(base_params, alt_min, alt_max, step):
    """
    Vary altitude from alt_min to alt_max in increments of
    step, holding all other params fixed.

    Returns a list of dicts:
        {'alt': float, 'CL': float, 'CD': float, 'XCP': float}
    """

    results = []

    step = step if step > 0 else 1000.0

    for alt in np.arange(alt_min, alt_max + step * 0.5, step):

        params = base_params.copy()
        params['alt'] = round(float(alt), 1)

        pred = aerodynamic_prediction(params)

        results.append({
            'alt': round(float(alt), 1),
            'CL' : pred['CL'],
            'CD' : pred['CD'],
            'XCP': pred['XCP'],
        })

    return results
