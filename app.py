# =========================================================
# app.py  –  DRDO Aerospace Optimization Platform  v2.0
# Professional UI — enlarged uniform outputs, aerospace theme
# =========================================================

import time
import PySimpleGUI as sg
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =========================================================
# BACKEND IMPORTS
# =========================================================
from predictor import aerodynamic_prediction
from optimizer import run_optimization
from envelope  import alpha_sweep, mach_sweep, altitude_sweep

# =========================================================
# AEROSPACE PRO THEME
# =========================================================
sg.LOOK_AND_FEEL_TABLE['AERO_PRO'] = {
    'BACKGROUND' : '#050A12',
    'TEXT'       : '#B8C4D0',
    'INPUT'      : '#0D1621',
    'TEXT_INPUT' : '#E8F4FF',
    'SCROLL'     : '#1B6FA8',
    'BUTTON'     : ('#FFFFFF', '#0D4F8B'),
    'PROGRESS'   : ('#00C9FF', '#050A12'),
    'BORDER'     : 1,
    'SLIDER_DEPTH'   : 0,
    'PROGRESS_DEPTH' : 0,
}
sg.theme('AERO_PRO')

# ── Uniform sizing constants ──────────────────────────────
LABEL_W   = 24
INPUT_W   = 13
OUT_W     = 26
BTN_W     = 20
BTN_H     = 1
COL_H     = 420
PLOT_W    = 960
PLOT_H    = 400
ML_H      = 16
ML_H_SM   =  6

# =========================================================
# PARAMETER BOUNDS
# =========================================================
default_bounds = {
    'nose_len'    : (120,   360),
    'body_len'    : (2400, 3000),
    'wing_le'     : (1000, 2000),
    'root_chord'  : (150,   250),
    'tip_chord'   : (110,   190),
    'semi_span'   : (600,  1500),
    'root_th'     : (15,    25),
    'tip_th'      : (5,     11),
    'wing_sweep'  : (0,     70),
    'tail_le'     : (2830, 2910),
    'root_chord1' : (80,   160),
    'tip_chord1'  : (30,    90),
    'semi_span1'  : (60,   140),
    'root_th1'    : (15,    21),
    'tip_th1'     : (5,     11),
    'mach'        : (0.2,   0.8),
    'alpha'       : (0,     20),
    'alt'         : (0,   6000),
}

params_list = list(default_bounds.keys())

LABELS = {
    'nose_len'    : 'Nose Length (mm)',
    'body_len'    : 'Body Length (mm)',
    'wing_le'     : 'Wing LE (mm)',
    'root_chord'  : 'Root Chord (mm)',
    'tip_chord'   : 'Tip Chord (mm)',
    'semi_span'   : 'Semi-Span (mm)',
    'root_th'     : 'Root Thickness',
    'tip_th'      : 'Tip Thickness',
    'wing_sweep'  : 'Wing Sweep (deg)',
    'tail_le'     : 'Tail LE (mm)',
    'root_chord1' : 'Tail Root Chord',
    'tip_chord1'  : 'Tail Tip Chord',
    'semi_span1'  : 'Tail Semi-Span',
    'root_th1'    : 'Tail Root Thickness',
    'tip_th1'     : 'Tail Tip Thickness',
    'mach'        : 'Mach Number',
    'alpha'       : 'Alpha (deg)',
    'alt'         : 'Altitude (m)',
}

# =========================================================
# COLOURS
# =========================================================
C_BG    = '#050A12'
C_PANEL = '#0D1621'
C_BORDER= '#1C2E44'
C_BLUE  = '#00A8FF'
C_GREEN = '#00E676'
C_RED   = '#FF4F5E'
C_AMBER = '#FFB300'
C_WHITE = '#E8F4FF'

# =========================================================
# FONT DEFINITIONS
# =========================================================
F_TITLE  = ('Consolas', 16, 'bold')
F_SUB    = ('Consolas', 11, 'bold')
F_LABEL  = ('Consolas',  9)
F_DATA   = ('Consolas',  9, 'bold')
F_MONO   = ('Courier New', 9)

# =========================================================
# HELPERS
# =========================================================

def safe_float(values, key, default=0.0):
    try:
        return float(values[key])
    except Exception:
        return default


def draw_figure(canvas_elem, figure):
    canvas = canvas_elem.TKCanvas
    try:
        for child in canvas.winfo_children():
            child.destroy()
    except Exception:
        pass
    fig_canvas = FigureCanvasTkAgg(figure, canvas)
    fig_canvas.draw()
    fig_canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
    return fig_canvas


def _section(title, color='#00A8FF'):
    return [sg.Text(
        f'── {title} ' + '─' * max(0, 55 - len(title)),
        font=('Consolas', 8, 'bold'), text_color=color,
        background_color=C_BG, pad=(0, 4)
    )]


def _out_row(label, key, unit=''):
    return [
        sg.Text(label, size=(LABEL_W, 1), font=F_LABEL,
                background_color=C_BG, text_color='#8AA0BA'),
        sg.Input('—', key=key, size=(OUT_W, 1), font=F_DATA,
                 text_color=C_GREEN, background_color=C_PANEL,
                 readonly=True, border_width=1,
                 disabled_readonly_background_color=C_PANEL,
                 disabled_readonly_text_color=C_GREEN),
        sg.Text(unit, size=(8, 1), font=F_LABEL,
                background_color=C_BG, text_color='#5A7A9A'),
    ]


# =========================================================
# STATUS BAR
# =========================================================
STATUS_BAR = [
    sg.Text('●', text_color=C_GREEN, background_color='#030608',
            font=('Consolas', 10), pad=(6, 2)),
    sg.Text('SYSTEM READY', key='STATUS', text_color=C_BLUE,
            background_color='#030608', font=('Consolas', 9, 'bold'),
            size=(50, 1), pad=(0, 2)),
    sg.Push(background_color='#030608'),
    sg.Text('DRDO / DRDL  ·  Aerodynamics Division',
            text_color='#3A5A7A', background_color='#030608',
            font=('Consolas', 8), pad=(6, 2)),
]

# =========================================================
# TAB 1 : PREDICTION
# =========================================================
pred_rows = []
for p in params_list:
    lo, hi = default_bounds[p]
    default = round((lo + hi) / 2, 2)
    pred_rows.append([
        sg.Text(LABELS[p], size=(LABEL_W, 1), font=F_LABEL,
                background_color=C_PANEL, text_color='#8AA0BA'),
        sg.Input(default, key=p, size=(INPUT_W, 1), font=F_DATA,
                 background_color='#0D1F33', text_color=C_WHITE,
                 border_width=1),
    ])

PRED_BTNS = [
    sg.Button('▶  ESTIMATE', key='Estimate',
              size=(BTN_W, BTN_H), font=F_SUB,
              button_color=('#FFFFFF', '#005F3B'), border_width=0),
    sg.Button('✕  CLEAR', key='Clear Prediction',
              size=(BTN_W, BTN_H), font=F_SUB,
              button_color=('#FFFFFF', '#5A1A22'), border_width=0),
]

# ── HIDDEN metric elements (invisible, zero-size, still in layout) ────────────
# These are kept so the event-loop can still call window[key].update()
# without KeyError — they just aren't visible to the user.
HIDDEN_METRICS = [
    sg.Input('—', key='MAE_OUT',  size=(0, 0), font=('Consolas', 1),
             visible=False, background_color=C_BG),
    sg.Input('—', key='RMSE_OUT', size=(0, 0), font=('Consolas', 1),
             visible=False, background_color=C_BG),
    sg.Input('—', key='R2_OUT',   size=(0, 0), font=('Consolas', 1),
             visible=False, background_color=C_BG),
    sg.Input('—', key='TIME_OUT', size=(0, 0), font=('Consolas', 1),
             visible=False, background_color=C_BG),
]

prediction_tab = [[sg.Column([
    _section('INPUT PARAMETERS', C_BLUE),
    [sg.Column(pred_rows,
               scrollable=True, vertical_scroll_only=True,
               size=(450, COL_H), background_color=C_PANEL,
               expand_y=True)],
    _section('ACTIONS'),
    [sg.Column([PRED_BTNS], background_color=C_BG, pad=(0, 6))],

    # ── Visible outputs: CL, CD, XCP only ────────────────
    _section('AERODYNAMIC COEFFICIENTS', C_GREEN),
    *[_out_row(*r) for r in [
        ('Lift Coefficient  CL',   'CL_OUT',   '—'),
        ('Drag Coefficient  CD',   'CD_OUT',   '—'),
        ('Centre of Pressure XCP', 'XCP_OUT',  'cal.'),
        ('Data Source',            'SRC_OUT',  ''),
    ]],

    # ── Hidden metric inputs (invisible, still addressable) ──
    [sg.Column([HIDDEN_METRICS],
               background_color=C_BG, pad=(0, 0), visible=False)],

    # NOTE: PLOT1 canvas is intentionally removed from the layout.
    # The bar-chart computation still runs internally in the event
    # loop but draw_figure() is not called, so nothing renders.

], background_color=C_BG, expand_x=True, expand_y=True,
   scrollable=False)
]]


# =========================================================
# TAB 2 : OPTIMIZATION
# =========================================================
bounds_rows = [[
    sg.Text('Parameter',    size=(LABEL_W, 1), font=F_LABEL,
            background_color=C_PANEL, text_color=C_AMBER),
    sg.Text('Lower Bound',  size=(INPUT_W, 1), font=F_LABEL,
            background_color=C_PANEL, text_color=C_AMBER),
    sg.Text('Upper Bound',  size=(INPUT_W, 1), font=F_LABEL,
            background_color=C_PANEL, text_color=C_AMBER),
]]
for p in params_list:
    lo, hi = default_bounds[p]
    bounds_rows.append([
        sg.Text(LABELS[p], size=(LABEL_W, 1), font=F_LABEL,
                background_color=C_PANEL, text_color='#8AA0BA'),
        sg.Input(lo, key=f'{p}_LOW',  size=(INPUT_W, 1),
                 font=F_DATA, background_color='#0D1F33',
                 text_color=C_WHITE, border_width=1),
        sg.Input(hi, key=f'{p}_HIGH', size=(INPUT_W, 1),
                 font=F_DATA, background_color='#0D1F33',
                 text_color=C_WHITE, border_width=1),
    ])

OPT_BTNS = [
    sg.Button('▶  RUN OPTIMIZATION', key='Run Optimization',
              size=(BTN_W + 4, BTN_H), font=F_SUB,
              button_color=('#FFFFFF', '#005F3B'), border_width=0),
    sg.Button('✕  CLEAR', key='Clear Optimization',
              size=(BTN_W, BTN_H), font=F_SUB,
              button_color=('#FFFFFF', '#5A1A22'), border_width=0),
]

optimization_tab = [[sg.Column([
    _section('PARAMETER BOUNDS', C_AMBER),
    [sg.Column(bounds_rows,
               scrollable=True, vertical_scroll_only=True,
               size=(550, COL_H), background_color=C_PANEL,
               expand_y=True)],

    _section('OUTPUT CONSTRAINTS  (dataset-valid)', C_RED),
    [sg.Text('CL  Min', size=(8,1), font=F_LABEL,
             background_color=C_BG, text_color='#8AA0BA'),
     sg.Input('-3.723',   key='CL_MIN',  size=(INPUT_W,1), font=F_DATA,
              background_color='#0D1F33', text_color=C_WHITE),
     sg.Text('CL  Max', size=(8,1), font=F_LABEL,
             background_color=C_BG, text_color='#8AA0BA'),
     sg.Input('15.2213',  key='CL_MAX',  size=(INPUT_W,1), font=F_DATA,
              background_color='#0D1F33', text_color=C_WHITE)],
    [sg.Text('CD  Min', size=(8,1), font=F_LABEL,
             background_color=C_BG, text_color='#8AA0BA'),
     sg.Input('-1.187',   key='CD_MIN',  size=(INPUT_W,1), font=F_DATA,
              background_color='#0D1F33', text_color=C_WHITE),
     sg.Text('CD  Max', size=(8,1), font=F_LABEL,
             background_color=C_BG, text_color='#8AA0BA'),
     sg.Input('5.7352',   key='CD_MAX',  size=(INPUT_W,1), font=F_DATA,
              background_color='#0D1F33', text_color=C_WHITE)],
    [sg.Text('XCP Min', size=(8,1), font=F_LABEL,
             background_color=C_BG, text_color='#8AA0BA'),
     sg.Input('-12.3114', key='XCP_MIN', size=(INPUT_W,1), font=F_DATA,
              background_color='#0D1F33', text_color=C_WHITE),
     sg.Text('XCP Max', size=(8,1), font=F_LABEL,
             background_color=C_BG, text_color='#8AA0BA'),
     sg.Input('-3.5322',  key='XCP_MAX', size=(INPUT_W,1), font=F_DATA,
              background_color='#0D1F33', text_color=C_WHITE)],

    _section('SOLVER SETTINGS', C_AMBER),
    [sg.Text('Population Size', size=(18,1), font=F_LABEL,
             background_color=C_BG, text_color='#8AA0BA'),
     sg.Input('15', key='POPSIZE', size=(INPUT_W,1), font=F_DATA,
              background_color='#0D1F33', text_color=C_WHITE),
     sg.Text('Max Iterations', size=(16,1), font=F_LABEL,
             background_color=C_BG, text_color='#8AA0BA'),
     sg.Input('50', key='MAXITER', size=(INPUT_W,1), font=F_DATA,
              background_color='#0D1F33', text_color=C_WHITE)],

    _section('ACTIONS'),
    [sg.Column([OPT_BTNS], background_color=C_BG, pad=(0, 6))],

    _section('OPTIMIZATION LOG', C_GREEN),
    [sg.Multiline(size=(120, ML_H), key='OPT_OUTPUT',
                  autoscroll=True, expand_x=True,
                  font=F_MONO,
                  background_color=C_PANEL, text_color=C_GREEN,
                  border_width=1)],

    _section('CONVERGENCE CHART', C_BLUE),
    [sg.Canvas(key='PLOT2', size=(PLOT_W, PLOT_H),
               expand_x=True, expand_y=True)],
], background_color=C_BG, expand_x=True, expand_y=True)
]]


# =========================================================
# TAB 3 : FLIGHT ENVELOPE
# =========================================================
def _sweep_frame(title, keys, defaults, key_color=C_BLUE):
    k_min, k_max, k_step = keys
    d_min, d_max, d_step = defaults
    return sg.Frame(title, [
        [sg.Text('Min',  size=(5,1), font=F_LABEL,
                 background_color=C_PANEL, text_color='#8AA0BA'),
         sg.Input(d_min,  key=k_min,  size=(INPUT_W,1), font=F_DATA,
                  background_color='#0D1F33', text_color=C_WHITE),
         sg.Text('Max',  size=(5,1), font=F_LABEL,
                 background_color=C_PANEL, text_color='#8AA0BA'),
         sg.Input(d_max,  key=k_max,  size=(INPUT_W,1), font=F_DATA,
                  background_color='#0D1F33', text_color=C_WHITE),
         sg.Text('Step', size=(5,1), font=F_LABEL,
                 background_color=C_PANEL, text_color='#8AA0BA'),
         sg.Input(d_step, key=k_step, size=(INPUT_W,1), font=F_DATA,
                  background_color='#0D1F33', text_color=C_WHITE)],
    ], font=('Consolas', 9, 'bold'), title_color=key_color,
       background_color=C_PANEL, border_width=1,
       relief=sg.RELIEF_FLAT)

FLT_BTNS = [
    sg.Button('▶  RUN FLIGHT ENVELOPE', key='Run Flight Envelope Analysis',
              size=(BTN_W + 6, BTN_H), font=F_SUB,
              button_color=('#FFFFFF', '#005F3B'), border_width=0),
    sg.Button('✕  CLEAR', key='Clear Analysis',
              size=(BTN_W, BTN_H), font=F_SUB,
              button_color=('#FFFFFF', '#5A1A22'), border_width=0),
]

flight_tab = [[sg.Column([
    _section('SWEEP CONFIGURATION', C_BLUE),
    [_sweep_frame('ALPHA SWEEP  (deg)',
                  ('ALPHA_MIN','ALPHA_MAX','ALPHA_STEP'),
                  ('0','20','2'), C_BLUE)],
    [_sweep_frame('MACH SWEEP',
                  ('MACH_MIN','MACH_MAX','MACH_STEP'),
                  ('0.2','0.8','0.1'), C_GREEN)],
    [_sweep_frame('ALTITUDE SWEEP  (m)',
                  ('ALT_MIN','ALT_MAX','ALT_STEP'),
                  ('0','6000','1000'), C_AMBER)],

    _section('ACTIONS'),
    [sg.Column([FLT_BTNS], background_color=C_BG, pad=(0, 6))],

    _section('SIMULATION RESULTS', C_GREEN),
    [sg.Multiline(size=(120, ML_H), key='FLIGHT_OUTPUT',
                  autoscroll=True, expand_x=True,
                  font=F_MONO,
                  background_color=C_PANEL, text_color=C_GREEN,
                  border_width=1)],

    _section('SUMMARY', C_AMBER),
    [sg.Multiline(size=(120, ML_H_SM), key='SUMMARY_OUTPUT',
                  expand_x=True, font=F_MONO,
                  background_color=C_PANEL, text_color=C_AMBER,
                  border_width=1)],

    _section('FLIGHT ENVELOPE CHART', C_BLUE),
    [sg.Canvas(key='PLOT3', size=(PLOT_W, PLOT_H),
               expand_x=True, expand_y=True)],
], background_color=C_BG, expand_x=True, expand_y=True)
]]


# =========================================================
# TAB GROUP
# =========================================================
TAB_STYLE = dict(
    expand_x=True, expand_y=True,
    pad=(0, 0),
)

tab_group = sg.TabGroup([[
    sg.Tab('  🔬  PREDICTION  ',     prediction_tab,  **TAB_STYLE,
           background_color=C_BG),
    sg.Tab('  ⚙  OPTIMIZATION  ',   optimization_tab, **TAB_STYLE,
           background_color=C_BG),
    sg.Tab('  ✈  FLIGHT ENVELOPE  ', flight_tab,      **TAB_STYLE,
           background_color=C_BG),
]],
    tab_location='top',
    font=('Consolas', 10, 'bold'),
    selected_title_color=C_WHITE,
    title_color='#5A7A9A',
    selected_background_color='#0D4F8B',
    background_color='#030608',
    tab_background_color='#030608',
    border_width=0,
    expand_x=True, expand_y=True,
    key='TABGROUP',
)

# =========================================================
# MAIN LAYOUT
# =========================================================
layout = [
    [sg.Column([
        [sg.Text(
            'AERODYNAMIC CONFIGURATION DESIGN & OPTIMIZATION',
            font=('Consolas', 18, 'bold'), text_color=C_BLUE,
            background_color='#030608', justification='center',
            expand_x=True, pad=(0, 4),
        )],
        [sg.Text(
            'DRDO  ·  Defence Research & Development Laboratory  ·  Digital Aerodynamics Twin',
            font=('Consolas', 9), text_color='#3A6080',
            background_color='#030608', justification='center',
            expand_x=True, pad=(0, 2),
        )],
    ], background_color='#030608', expand_x=True, pad=(0, 0))],

    [tab_group],

    [sg.Column([STATUS_BAR], background_color='#030608',
               expand_x=True, pad=(0, 0))],
]

window = sg.Window(
    'DRDO Aerospace Optimization Platform  v2.0',
    layout,
    size=(1600, 980),
    finalize=True,
    resizable=True,
    element_justification='left',
    background_color=C_BG,
    margins=(0, 0),
)
window.Maximize()


# =========================================================
# PLOT STYLE HELPERS
# =========================================================

def _style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(C_PANEL)
    ax.set_title(title,   color=C_WHITE, fontsize=10, fontweight='bold',
                 fontfamily='Consolas', pad=8)
    ax.set_xlabel(xlabel, color='#8AA0BA', fontsize=8, fontfamily='Consolas')
    ax.set_ylabel(ylabel, color='#8AA0BA', fontsize=8, fontfamily='Consolas')
    ax.tick_params(colors='#6A8BA0', labelsize=7)
    ax.spines[:].set_color(C_BORDER)
    ax.grid(True, color=C_BORDER, alpha=0.5, linewidth=0.6)


def _make_fig(cols=1, rows=1, w=10, h=4):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor(C_BG)
    plt.tight_layout(pad=2.5)
    return fig, axes


# =========================================================
# STATUS HELPER
# =========================================================

def set_status(msg, color=C_BLUE):
    window['STATUS'].update(msg)
    window['STATUS'].Widget.config(fg=color)
    window.refresh()


# =========================================================
# EVENT LOOP
# =========================================================
while True:

    event, values = window.read()

    if event in (sg.WINDOW_CLOSED, 'Exit'):
        break

    # =================================================
    # TAB 1 : PREDICTION
    # =================================================
    if event == 'Estimate':
        set_status('⏳  Running aerodynamic prediction …', C_AMBER)
        t0 = time.perf_counter()

        params = {p: safe_float(values, p) for p in params_list}

        try:
            result  = aerodynamic_prediction(params)
            elapsed = round(time.perf_counter() - t0, 3)

            # ── Visible outputs ───────────────────────────
            window['CL_OUT' ].update(f"{result.get('CL',  '—')}")
            window['CD_OUT' ].update(f"{result.get('CD',  '—')}")
            window['XCP_OUT'].update(f"{result.get('XCP', '—')}")
            window['SRC_OUT'].update(result.get('source', '—').upper())

            # ── Hidden metric updates (internal only) ─────
            # Values are stored in invisible Input elements so the
            # rest of the pipeline can still read them if needed.
            m = result.get('metrics', {})
            window['MAE_OUT' ].update(m.get('MAE',  '—'))
            window['RMSE_OUT'].update(m.get('RMSE', '—'))
            window['R2_OUT'  ].update(m.get('R2',   '—'))
            window['TIME_OUT'].update(f"{elapsed}")

            # ── Bar chart computed internally, not rendered ──
            # The figure is built so downstream logic (e.g. export
            # or logging) can use it, but draw_figure() is NOT
            # called — no canvas exists for PLOT1 in this layout.
            fig, axes = _make_fig(cols=3, w=10, h=4)
            coeff_names = ['CL', 'CD', 'XCP']
            coeff_vals  = [result.get('CL',  0),
                           result.get('CD',  0),
                           result.get('XCP', 0)]
            bar_colors  = [C_BLUE, C_RED, C_GREEN]

            for ax, name, val, col in zip(axes, coeff_names,
                                          coeff_vals, bar_colors):
                bar = ax.bar([name], [val], color=col, width=0.45,
                             edgecolor='none')
                ax.bar_label(bar, labels=[f'{val:.4f}'],
                             color=col, fontsize=9,
                             fontfamily='Consolas', padding=4)
                _style_ax(ax, name, '', 'Value')

            fig.suptitle(
                'Predicted Aerodynamic Coefficients',
                color=C_BLUE, fontsize=11, fontweight='bold',
                fontfamily='Consolas', y=1.01
            )
            plt.tight_layout()
            # draw_figure(window['PLOT1'], fig)  ← intentionally
            #   removed; PLOT1 canvas is not in the layout.
            plt.close(fig)   # free memory — figure computed, not shown

            set_status(
                f'✔  Prediction complete  |  Source: '
                f'{result.get("source","").upper()}  |  '
                f'Elapsed: {elapsed} s', C_GREEN
            )

        except Exception as e:
            set_status(f'✘  Prediction Error: {e}', C_RED)
            sg.popup_error(f'Prediction Error:\n{e}')

    if event == 'Clear Prediction':
        for key in ['CL_OUT','CD_OUT','XCP_OUT','SRC_OUT',
                    'MAE_OUT','RMSE_OUT','R2_OUT','TIME_OUT']:
            window[key].update('—')
        set_status('SYSTEM READY')

    # =================================================
    # TAB 2 : OPTIMIZATION
    # =================================================
    if event == 'Run Optimization':
        set_status('⏳  Differential Evolution running …', C_AMBER)
        t0 = time.perf_counter()

        try:
            bounds = [
                (safe_float(values, f'{p}_LOW'),
                 safe_float(values, f'{p}_HIGH'))
                for p in params_list
            ]
            constraints = {
                'CL'  : (safe_float(values, 'CL_MIN'),
                         safe_float(values, 'CL_MAX')),
                'CD'  : (safe_float(values, 'CD_MIN'),
                         safe_float(values, 'CD_MAX')),
                'XCP' : (safe_float(values, 'XCP_MIN'),
                         safe_float(values, 'XCP_MAX')),
            }

            result, conv_hist = run_optimization(
                bounds,
                maxiter     = int(safe_float(values, 'MAXITER', 50)),
                popsize     = int(safe_float(values, 'POPSIZE', 15)),
                constraints = constraints,
            )

            elapsed   = round(time.perf_counter() - t0, 2)
            best_x    = result.x
            best_prm  = {p: round(float(best_x[i]), 4)
                         for i, p in enumerate(params_list)}
            best_pred = aerodynamic_prediction(best_prm)

            lines  = ['═' * 68,
                      ' OPTIMIZATION  COMPLETED',
                      '═' * 68, '']
            lines += [f'  Best CL/CD Ratio : {-result.fun:.6f}',
                      f'  Best CL          : {best_pred["CL"]}',
                      f'  Best CD          : {best_pred["CD"]}',
                      f'  Best XCP         : {best_pred["XCP"]}',
                      f'  Elapsed          : {elapsed} s',
                      '']
            lines += ['─' * 68,
                      '  OPTIMAL GEOMETRY & FLIGHT CONDITIONS',
                      '─' * 68]
            for p, v in best_prm.items():
                lines.append(f'  {LABELS[p]:<32}: {v}')
            lines += ['', '═' * 68,
                      '  CONVERGENCE HISTORY',
                      '═' * 68, '']
            for item in conv_hist:
                lines.append(
                    f"  Gen {item['generation']:>4d}  │  "
                    f"Fitness : {item['fitness']:.8f}"
                )

            window['OPT_OUTPUT'].update('\n'.join(lines))

            # ── Convergence plot ─────────────────────────
            gens = [h['generation'] for h in conv_hist]
            fits = [h['fitness']    for h in conv_hist]

            fig, ax = _make_fig(w=10, h=4)
            ax.plot(gens, fits, marker='o', linewidth=2,
                    color=C_BLUE, markersize=4,
                    markerfacecolor=C_AMBER, markeredgewidth=0)
            ax.fill_between(gens, fits, alpha=0.08, color=C_BLUE)
            _style_ax(ax, 'Optimization Convergence',
                      'Generation', 'Convergence Value')
            plt.tight_layout()
            draw_figure(window['PLOT2'], fig)
            plt.close(fig)

            set_status(
                f'✔  Optimization complete  |  '
                f'Best CL/CD = {-result.fun:.4f}  |  '
                f'Elapsed: {elapsed} s', C_GREEN
            )

        except Exception as e:
            set_status(f'✘  Optimization Error: {e}', C_RED)
            sg.popup_error(f'Optimization Error:\n{e}')

    if event == 'Clear Optimization':
        window['OPT_OUTPUT'].update('')
        set_status('SYSTEM READY')

    # =================================================
    # TAB 3 : FLIGHT ENVELOPE
    # =================================================
    if event == 'Run Flight Envelope Analysis':
        set_status('⏳  Running flight envelope sweeps …', C_AMBER)
        t0 = time.perf_counter()

        params = {p: safe_float(values, p) for p in params_list}

        try:
            alpha_res = alpha_sweep(
                params,
                safe_float(values, 'ALPHA_MIN'),
                safe_float(values, 'ALPHA_MAX'),
                safe_float(values, 'ALPHA_STEP'),
            )
            mach_res = mach_sweep(
                params,
                safe_float(values, 'MACH_MIN'),
                safe_float(values, 'MACH_MAX'),
                safe_float(values, 'MACH_STEP'),
            )
            alt_res = altitude_sweep(
                params,
                safe_float(values, 'ALT_MIN'),
                safe_float(values, 'ALT_MAX'),
                safe_float(values, 'ALT_STEP'),
            )

            elapsed = round(time.perf_counter() - t0, 2)

            HDR = f'{"─"*54}'
            out  = ['═'*54, ' ALPHA SWEEP', '═'*54]
            out += [f'  {"Alpha":>8}  {"CL":>10}  {"CD":>10}  {"XCP":>12}', HDR]
            for r in alpha_res:
                out.append(
                    f'  {r["alpha"]:>8.2f}  {r["CL"]:>10.4f}'
                    f'  {r["CD"]:>10.4f}  {r["XCP"]:>12.4f}'
                )
            out += ['', '═'*54, ' MACH SWEEP', '═'*54]
            out += [f'  {"Mach":>8}  {"CL":>10}  {"CD":>10}  {"XCP":>12}', HDR]
            for r in mach_res:
                out.append(
                    f'  {r["mach"]:>8.3f}  {r["CL"]:>10.4f}'
                    f'  {r["CD"]:>10.4f}  {r["XCP"]:>12.4f}'
                )
            out += ['', '═'*54, ' ALTITUDE SWEEP', '═'*54]
            out += [f'  {"Alt (m)":>10}  {"CL":>10}  {"CD":>10}  {"XCP":>12}', HDR]
            for r in alt_res:
                out.append(
                    f'  {r["alt"]:>10.1f}  {r["CL"]:>10.4f}'
                    f'  {r["CD"]:>10.4f}  {r["XCP"]:>12.4f}'
                )

            window['FLIGHT_OUTPUT'].update('\n'.join(out))
            window['SUMMARY_OUTPUT'].update(
                f'  Alpha Cases    : {len(alpha_res)}\n'
                f'  Mach  Cases    : {len(mach_res)}\n'
                f'  Altitude Cases : {len(alt_res)}\n'
                f'  Total Sims     : {len(alpha_res)+len(mach_res)+len(alt_res)}'
                f'    |    Elapsed : {elapsed} s'
            )

            fig, axes = _make_fig(cols=3, w=14, h=4)

            ax = axes[0]
            ax.plot([r['alpha'] for r in alpha_res],
                    [r['CL']    for r in alpha_res],
                    marker='o', linewidth=2, color=C_BLUE,
                    markersize=4, markerfacecolor=C_WHITE)
            ax.fill_between([r['alpha'] for r in alpha_res],
                            [r['CL']    for r in alpha_res],
                            alpha=0.07, color=C_BLUE)
            _style_ax(ax, 'CL  vs  Alpha', 'Alpha (deg)', 'CL')

            ax = axes[1]
            ax.plot([r['mach'] for r in mach_res],
                    [r['CL']   for r in mach_res],
                    marker='s', linewidth=2, color=C_GREEN,
                    markersize=4, markerfacecolor=C_WHITE)
            ax.fill_between([r['mach'] for r in mach_res],
                            [r['CL']   for r in mach_res],
                            alpha=0.07, color=C_GREEN)
            _style_ax(ax, 'CL  vs  Mach', 'Mach', 'CL')

            ax = axes[2]
            ax.plot([r['alt'] for r in alt_res],
                    [r['CD']  for r in alt_res],
                    marker='^', linewidth=2, color=C_RED,
                    markersize=4, markerfacecolor=C_WHITE)
            ax.fill_between([r['alt'] for r in alt_res],
                            [r['CD']  for r in alt_res],
                            alpha=0.07, color=C_RED)
            _style_ax(ax, 'CD  vs  Altitude', 'Altitude (m)', 'CD')

            plt.tight_layout()
            draw_figure(window['PLOT3'], fig)
            plt.close(fig)

            set_status(
                f'✔  Flight envelope complete  |  '
                f'{len(alpha_res)+len(mach_res)+len(alt_res)} sims  |  '
                f'Elapsed: {elapsed} s', C_GREEN
            )

        except Exception as e:
            set_status(f'✘  Flight Envelope Error: {e}', C_RED)
            sg.popup_error(f'Flight Envelope Error:\n{e}')

    if event == 'Clear Analysis':
        window['FLIGHT_OUTPUT'].update('')
        window['SUMMARY_OUTPUT'].update('')
        set_status('SYSTEM READY')

# =========================================================
# CLOSE
# =========================================================
window.close()