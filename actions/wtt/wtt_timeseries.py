#!/usr/bin/env python3
"""
Whole-tile-trigger (WTT) time series from the json written by wtt.py

Panels, sharing the time axis:
  1. WTT events per hour of run time: per file, rolling over the last
     --n_rolling files, and per calendar day (CDT)
  2. percentage of beam spills with a WTT, the same three ways
A file is 10 min of data with mostly 0-2 WTTs, so per-file values are
quantised (1 WTT / file = 6 / hour)

Writes OUTPUT (png), OUTPUT_last.png (last 50 files) and OUTPUT.html (plotly).
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

INK, GRID = '#3b3b38', '#d9d8d2'
REF_C = '#6b6a64'

# Run 1 reference: totals over the v11 beam reflow files from 2024-07-09 00:00 CDT
# to the last file (2024-07-12 03:50 CDT; 429 files, 71.4 h), calib_prompt_hits
# and the default cuts in wtt_funcs.py: 220 WTT events in 71.4 h, and 109 of
# 197081 beam spills with a WTT.
RUN1_LABEL = 'Run 1 average (Jul 9-12 2024)'
RUN1_WTT_EVENTS_PER_HOUR = 3.08
RUN1_BEAM_SPILLS_WITH_WTT_PCT = 0.0553


def gehrels(k):
    """(lower, upper) 1-sigma error on Poisson counts k (arrays)."""
    k = np.asarray(k, dtype=float)
    up = np.sqrt(k + 0.75) + 1.0
    lo = np.where(k > 0, np.sqrt(np.clip(k - 0.25, 0, None)), 0.0)
    return lo, up


def load(input_file):
    with open(input_file) as f:
        entries = json.load(f)["wtt"]
    entries.sort(key=lambda e: e["timestamp"])
    t = [datetime.fromisoformat(e["timestamp"]) for e in entries]
    hours = np.array([max(e["run_time_s"] or 1, 1) / 3600.0 for e in entries])
    n_wtt = np.array([e["n_wtt_events"] for e in entries])
    n_beam = np.array([e["n_beam_events"] for e in entries])
    n_wtt_beam = np.array([e["n_wtt_beam_events"] for e in entries])
    files = [e["file"] for e in entries]
    return t, hours, n_wtt, n_beam, n_wtt_beam, files


def daily(t, num, den):
    """Per-calendar-day totals (in the files' own UTC offset, i.e. CDT):
    (mean file start time of each day, sum(num), sum(den))."""
    days = {}
    for x, n, d in zip(t, num, den):
        a = days.setdefault(x.date(), [[], 0.0, 0.0])
        a[0].append(x)
        a[1] += n
        a[2] += d
    keys = sorted(days)
    centres = [datetime.fromtimestamp(np.mean([x.timestamp() for x in days[k][0]]), tz=days[k][0][0].tzinfo)
               for k in keys]
    return centres, np.array([days[k][1] for k in keys]), np.array([days[k][2] for k in keys])


def rolling(t, num, den, n, max_gap_s):
    """sum(num) / sum(den) over the last n files, restarting after any gap of
    more than max_gap_s between consecutive files; NaN until the window holds
    at least n // 2 files, so a restart does not draw a one-file spike.
    Returns (x, y) with a NaN point at each gap so the line breaks there."""
    xs, ys = [], []
    seg_start = 0
    for i in range(len(t)):
        if i > 0 and (t[i] - t[i - 1]).total_seconds() > max_gap_s:
            seg_start = i
            xs.append(t[i - 1])
            ys.append(np.nan)
        lo = max(seg_start, i - n + 1)
        d = den[lo:i + 1].sum()
        xs.append(t[i])
        enough = (i + 1 - lo) >= max(1, n // 2)
        ys.append(num[lo:i + 1].sum() / d if (enough and d > 0) else np.nan)
    return xs, np.array(ys)


def series(t, hours, n_wtt, n_beam, n_wtt_beam, n_rolling=12, max_gap_s=1800):
    """[(panel title, y label, [item, ...])]; an item is
    ('points', label, colour, marker, size, alpha, x, y, yerr_lo, yerr_hi),
    with yerr None for no error bars, ('line', label, colour, x, y), or
    ('ref', label, colour, value) for a horizontal reference line."""
    file_c, roll_c, day_c = '#9ec5f4', '#2a78d6', '#184f95'   # light / mid / dark blue ramp steps
    roll_label = 'rolling, %d files' % n_rolling

    rx, ry = rolling(t, n_wtt, hours, n_rolling, max_gap_s)
    dx, dn, dh = daily(t, n_wtt, hours)
    dlo, dup = gehrels(dn)
    rate = [('points', 'per file', file_c, 'o', 3, 0.7, t, n_wtt / hours, None, None),
            ('line', roll_label, roll_c, rx, ry),
            ('points', 'per day', day_c, 'D', 7, 1.0, dx, dn / dh, dlo / dh, dup / dh)]

    has_beam = n_beam > 0
    tb = [x for x, keep in zip(t, has_beam) if keep]
    k, nb = n_wtt_beam[has_beam], n_beam[has_beam]
    rx, ry = rolling(tb, k, nb, n_rolling, max_gap_s)
    dx, dk, dnb = daily(tb, k, nb)
    dlo, dup = gehrels(dk)
    beam = [('points', 'per file', file_c, 'o', 3, 0.7, tb, 100.0 * k / nb, None, None),
            ('line', roll_label, roll_c, rx, 100.0 * ry),
            ('points', 'per day', day_c, 'D', 7, 1.0, dx, 100.0 * dk / dnb, 100.0 * dlo / dnb, 100.0 * dup / dnb)]
    rate.append(('ref', RUN1_LABEL, REF_C, RUN1_WTT_EVENTS_PER_HOUR))
    beam.append(('ref', RUN1_LABEL, REF_C, RUN1_BEAM_SPILLS_WITH_WTT_PCT))
    return [('WTT events per hour', 'WTT events / hour', rate),
            ('Beam spills with a WTT', 'beam spills with WTT [%]', beam)]


def crop(panels, t0):
    """Keep only what is at or after t0. Rolling and daily values keep their
    full-history windows."""
    out = []
    for name, ylabel, items in panels:
        new = []
        for it in items:
            if it[0] == 'ref':
                new.append(it)
            elif it[0] == 'points':
                keep = np.array([x >= t0 for x in it[6]], dtype=bool)
                sel = lambda a: None if a is None else np.asarray(a)[keep]
                new.append(it[:6] + ([x for x, k in zip(it[6], keep) if k], sel(it[7]), sel(it[8]), sel(it[9])))
            else:
                keep = [x >= t0 for x in it[3]]
                new.append(it[:3] + ([x for x, k in zip(it[3], keep) if k],
                                     np.asarray(it[4])[np.array(keep, dtype=bool)]))
        out.append((name, ylabel, new))
    return out


def plot_png(panels, output, title):
    matplotlib.rcParams['timezone'] = 'US/Central'
    fig, axs = plt.subplots(len(panels), 1, figsize=(11, 6.5), sharex=True)
    for ax, (name, ylabel, items) in zip(axs, panels):
        for it in items:
            if it[0] == 'ref':
                _, label, colour, value = it
                ax.axhline(value, color=colour, lw=1.5, ls='--', label='%s: %.3g' % (label, value), zorder=1)
            elif it[0] == 'points':
                _, label, colour, marker, size, alpha, x, y, elo, eup = it
                ax.errorbar(x, y, yerr=None if elo is None else [elo, eup], fmt=marker, ms=size,
                            color=colour, mec='white', mew=0.5, ecolor=colour, elinewidth=1,
                            capsize=0, alpha=alpha, label=label, ls='none', zorder=3)
            else:
                _, label, colour, x, y = it
                ax.plot(x, y, color=colour, lw=2, label=label, zorder=2)
        if all(len(it[6] if it[0] == 'points' else it[3]) == 0 for it in items if it[0] != 'ref'):
            # e.g. the beam panel for cosmic-only runs
            ax.text(0.5, 0.25, 'no %s in these files' % ('beam spills' if 'beam' in name.lower() else 'data'),
                    transform=ax.transAxes, ha='center', va='center', color=INK, fontsize=10)
        ax.set_title(name, loc='left', fontsize=10, color=INK)
        ax.set_ylabel(ylabel, color=INK)
        ax.set_ylim(bottom=0)
        refs = [it[3] for it in items if it[0] == 'ref']
        if refs:   # keep the reference line clear of the legend, e.g. on an empty panel
            ax.set_ylim(top=max(ax.get_ylim()[1], 2.0 * max(refs)))
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=8, ncol=len(items), loc='upper left', frameon=False)
    axs[-1].set_xlabel("File start [CDT]", color=INK)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y\n%H:%M CDT"))
    fig.suptitle(title, x=0.01, ha='left', color=INK)
    fig.tight_layout()
    fig.savefig(output, dpi=110)
    plt.close(fig)


def plot_html(panels, files, output, title):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[p[0] for p in panels])
    plotly_marker = {'o': 'circle', 's': 'square', '^': 'triangle-up', 'D': 'diamond'}
    for row, (name, ylabel, items) in enumerate(panels, start=1):
        for it in items:
            if it[0] == 'ref':
                _, label, colour, value = it
                fig.add_hline(y=value, line=dict(color=colour, width=1.5, dash='dash'),
                              annotation_text='%s: %.3g' % (label, value), annotation_position='top left',
                              annotation_font=dict(size=10, color=colour), row=row, col=1)
            elif it[0] == 'points':
                _, label, colour, marker, size, alpha, x, y, elo, eup = it
                per_file = len(x) == len(files)
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode='markers', name='%s: %s' % (name, label), opacity=alpha,
                    marker=dict(color=colour, symbol=plotly_marker[marker], size=size + 3,
                                line=dict(color='white', width=1)),
                    error_y=None if elo is None else dict(type='data', symmetric=False, array=eup,
                                                          arrayminus=elo, color=colour, thickness=1, width=0),
                    text=files if per_file else None,
                    hovertemplate='%{x|%m/%d/%Y %H:%M}<br>' + label + ': %{y:.3g}'
                                  + ('<br>%{text}' if per_file else '') + '<extra></extra>'),
                    row=row, col=1)
            else:
                _, label, colour, x, y = it
                fig.add_trace(go.Scatter(
                    x=x, y=[None if np.isnan(v) else v for v in y], mode='lines',
                    name='%s: %s' % (name, label), line=dict(color=colour, width=2),
                    hovertemplate='%{x|%m/%d/%Y %H:%M}<br>' + label + ': %{y:.3g}<extra></extra>'),
                    row=row, col=1)
        fig.update_yaxes(title_text=ylabel, rangemode='tozero', row=row, col=1)
    fig.update_xaxes(tickformat="%m/%d/%Y<br>%H:%M CDT")
    fig.update_layout(title=title, template="plotly_white", width=1100, height=650,
                      hovermode='closest')
    fig.write_html(output)


def main(input_file, output_file, n_last=50, n_rolling=12, max_gap_min=30.0):
    t, hours, n_wtt, n_beam, n_wtt_beam, files = load(input_file)
    if not t:
        print("No entries in", input_file)
        return
    base = output_file[:-4] if output_file.endswith('.png') else output_file

    panels = series(t, hours, n_wtt, n_beam, n_wtt_beam,
                    n_rolling=n_rolling, max_gap_s=60.0 * max_gap_min)
    plot_png(panels, output_file, "Whole-tile-trigger time series")
    plot_png(crop(panels, t[-n_last] if len(t) > n_last else t[0]), base + "_last.png",
             "Whole-tile-trigger time series, last %d files" % n_last)
    plot_html(panels, files, base + ".html", "Whole-tile-trigger time series")
    print("Wrote", output_file, base + "_last.png", base + ".html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_file', required=True, help="WTT json written by wtt.py")
    parser.add_argument('--output_file', required=True, help="Output plot (png)")
    parser.add_argument('--n_last', type=int, default=50)
    parser.add_argument('--n_rolling', type=int, default=12,
                        help='files in the rolling rate (default 12 = 2 h of 10 min files)')
    parser.add_argument('--max_gap_min', type=float, default=30.0,
                        help='restart the rolling window after a gap this long between files')
    args = parser.parse_args()
    print(args)
    main(**vars(args))
