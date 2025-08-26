#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEM (Excitation-Emission Matrix) interactive viewer
Standalone Python script that converts a CSV to an interactive HTML file.
No browser preview is triggered automatically.

Dependencies:
  pip install numpy pandas plotly

Usage:
  python eem_viewer.py /path/to/data.csv

Optional args:
  --rayleigh-margin 0.0   # mask width around Rayleigh line in nm
"""

import os
import argparse
from math import floor
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ---------- Defaults (edit or override via CLI) ----------
EM_Start = None   # emission min (X) in nm
EM_End   = None   # emission max (X) in nm
EX_Start = None   # excitation min (Y) in nm
EX_End   = None   # excitation max (Y) in nm

# Figure geometry
FIG_W, FIG_H = 1600, 800
HX = [0.06, 0.70]   # heatmap x-domain
HY = [0.12, 0.93]   # heatmap y-domain
PX = [0.75, 1.00]   # profile x-domain
PY = [0.45, 0.85]   # profile y-domain

# Labels
X_LABEL = "Emission wavelength [nm]"
Y_LABEL = "Excitation wavelength [nm]"
Z_LABEL = "Intensity (norm)"


def _norm_profile(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    m = np.nanmax(arr) if np.isfinite(np.nanmax(arr)) else 0.0
    if m > 0:
        out = arr / m
        out[np.isnan(arr)] = np.nan
        return out
    return np.where(np.isnan(arr), np.nan, 0.0).astype(np.float32)


def _discrete_rgb_scale(n_major=10, n_minor=3):
    """
    Discrete rainbow (no black floor): Blue → Cyan → Green → Yellow → Orange → Red.
    n_major × n_minor flat bands (no blending). Defaults: 10×3 = 30 bands.
    """
    anchors = np.array([
        (  0,   0, 255),   # blue
        (  0, 255, 255),   # cyan
        (  0, 255,   0),   # green
        (255, 255,   0),   # yellow
        (255, 165,   0),   # orange
        (255,   0,   0),   # red
    ], dtype=float)

    total = n_major * n_minor
    pos = np.linspace(0, len(anchors)-1, total)
    low = np.floor(pos).astype(int)
    high = np.clip(low + 1, 0, len(anchors)-1)
    t = pos - low
    cols = ((1 - t)[:, None] * anchors[low] + t[:, None] * anchors[high]).astype(int)

    scale = []
    for i, (r, g, b) in enumerate(cols):
        c = f"rgb({r},{g},{b})"
        a = i / total
        bnd = (i + 1) / total
        scale.append([a, c])
        scale.append([bnd, c])
    # force exact red at 1.0
    scale[-1][0] = 1.0
    scale[-1][1] = "rgb(255,0,0)"
    return scale


def render_one(csv_path: str, rayleigh_margin_nm: float = 0.0):
    out_path = os.path.splitext(csv_path)[0] + ".html"

    # --- Read CSV (col0=Emission, col1=Excitation, col2=Intensity) ---
    df = pd.read_csv(csv_path, skiprows=2, header=None, usecols=[0, 1, 2])
    df.columns = ["em", "exc", "int"]
    for c in ["em", "exc", "int"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["em", "exc"])
    if df.empty:
        raise ValueError("No valid numeric (em, exc) rows after parsing.")

    # Axes from data
    x_vals = np.sort(df["em"].unique()).astype(np.float32)   # X = emission
    y_vals = np.sort(df["exc"].unique()).astype(np.float32)  # Y = excitation
    x_min_data, x_max_data = float(x_vals.min()), float(x_vals.max())
    y_min_data, y_max_data = float(y_vals.min()), float(y_vals.max())

    # Apply user ranges if provided
    x_min = float(EM_Start) if EM_Start is not None else x_min_data
    x_max = float(EM_End)   if EM_End   is not None else x_max_data
    y_min = float(EX_Start) if EX_Start is not None else y_min_data
    y_max = float(EX_End)   if EX_End   is not None else y_max_data

    # Pivot to grid (rows=exc, cols=em), duplicates averaged
    Z_raw_df = (df.pivot_table(index="exc", columns="em", values="int", aggfunc="mean")
                  .reindex(index=y_vals, columns=x_vals))
    Z_raw = Z_raw_df.to_numpy(dtype=np.float32, copy=False)

    # Clamp negatives BEFORE masking/normalization
    Z_raw = np.maximum(Z_raw, 0.0)

    # Mask first: EX ≥ EM − margin  →  Y ≥ X − margin
    Xg, Yg = np.meshgrid(x_vals, y_vals)
    bad = (Yg >= (Xg - rayleigh_margin_nm))
    Z_masked = Z_raw.copy()
    Z_masked[bad] = np.nan

    # Normalize on valid region only
    max_val = np.nanmax(Z_masked) if np.isfinite(np.nanmax(Z_masked)) else 0.0
    Z_plot = (Z_masked / max_val).astype(np.float32) if max_val > 0 else np.zeros_like(Z_masked)

    # Gray overlay for masked region
    tri_overlay = np.where(bad, 1.0, np.nan).astype(np.float32)

    # Profile panel X limits = union of chosen EM/EX limits
    profile_xmin, profile_xmax = float(min(x_min, y_min)), float(max(x_max, y_max))

    # Default crosshair at midpoints of data arrays
    ix0, iy0 = len(x_vals)//2, len(y_vals)//2
    x0, y0 = float(x_vals[ix0]), float(y_vals[iy0])

    # ---------- Figure ----------
    fig = go.Figure()

    # Heatmap with discrete RGB-style scale and boxed colorbar
    discrete_scale = _discrete_rgb_scale(n_major=10, n_minor=3)
    cb_x = 0.5 * (HX[0] + HX[1])
    fig.add_trace(go.Heatmap(
        x=x_vals, y=y_vals, z=Z_plot,
        colorscale=discrete_scale,
        zmin=0, zmax=1,
        colorbar=dict(
            title=Z_LABEL, orientation="h",
            x=cb_x, y=0.0,
            len=max(0.0, (HX[1]-HX[0])-0.04), thickness=14,
            xanchor="center", yanchor="top",
            outlinecolor="black", outlinewidth=2  # boxed colorbar
        ),
        xaxis="x", yaxis="y", uid="heat",
        hovertemplate=f"{X_LABEL}: %{{x}}<br>{Y_LABEL}: %{{y}}<br>{Z_LABEL}: %{{z:.3f}}<extra></extra>"
    ))

    # Solid black contour isolines only (no fill), 9 levels: 0.1..0.9
    fig.add_trace(go.Contour(
        x=x_vals, y=y_vals, z=Z_plot,
        contours=dict(start=0.1, end=0.9, size=0.1, coloring="none", showlabels=False),
        line=dict(width=2, color="black"),
        showscale=False, hoverinfo="skip",
        xaxis="x", yaxis="y", uid="contours"
    ))

    # Gray invalid overlay
    fig.add_trace(go.Heatmap(
        x=x_vals, y=y_vals, z=tri_overlay,
        colorscale=[[0.0, "lightgray"], [1.0, "lightgray"]],
        showscale=False, hoverinfo="skip", opacity=0.6,
        xaxis="x", yaxis="y", uid="mask-tri"
    ))

    # Crosshair shapes
    fig.add_shape(type="line", x0=x_min, x1=x_max, y0=y0, y1=y0,
                  line=dict(color="black", width=1.5, dash="dash"), xref="x", yref="y")
    fig.add_shape(type="line", x0=x0, x1=x0, y0=y_min, y1=y_max,
                  line=dict(color="black", width=1.5, dash="dash"), xref="x", yref="y")

    # Initial profiles (independent normalization)
    row0 = _norm_profile(Z_plot[iy0, :])
    col0 = _norm_profile(Z_plot[:, ix0])
    fig.add_trace(go.Scatter(
        x=x_vals, y=row0, mode="lines",
        line=dict(width=3, color="tomato"),
        connectgaps=False, showlegend=False, uid="prof_em",
        xaxis="x2", yaxis="y2"
    ))
    fig.add_trace(go.Scatter(
        x=y_vals, y=col0, mode="lines",
        line=dict(width=3, color="dodgerblue"),
        connectgaps=False, showlegend=False, uid="prof_ex",
        xaxis="x2", yaxis="y2"
    ))

    # "EX ≥ EM" label centered in masked region
    if np.any(bad):
        iy_bad, ix_bad = np.where(bad)
        cx = float(np.mean(Xg[iy_bad, ix_bad])); cy = float(np.mean(Yg[iy_bad, ix_bad]))
    else:
        m = 0.5 * (max(x_min, y_min) + min(x_max, y_max)); cx, cy = m, m
    fig.add_annotation(x=cx, y=cy, xref="x", yref="y", text="EX ≥ EM", showarrow=False,
                       font=dict(color="black", size=36), xanchor="center", yanchor="middle", opacity=0.95)

    # Big title = filename (3×, bold, black)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    fig.add_annotation(
        x=0.5, y=1.00, xref="paper", yref="paper",
        text=base, showarrow=False,
        font=dict(size=48, color="black", family="Arial Black"),
        xanchor="center", yanchor="bottom"
    )

    # Profile title annotation (above profile panel)
    title_x = (PX[0] + PX[1]) / 2.0
    title_y = PY[1] + 0.02
    fig.add_annotation(
        x=title_x, y=title_y, xref="paper", yref="paper",
        xanchor="center", yanchor="bottom",
        text=f"Emission@{x0:.0f} nm | Excitation@{y0:.0f} nm",
        showarrow=False, font=dict(color="black", size=13)
    )

    # Layout (profile gridlines 50% black; X ticks every 50 nm; Y ticks 0.2 with 1 decimal; Y fixed [0,1])
    fig.update_layout(
        xaxis=dict(domain=HX, range=[x_min, x_max], title=X_LABEL,
                   fixedrange=True, showgrid=False, showline=True, mirror=True,
                   linecolor="lightgray", linewidth=1, ticks="outside", ticklen=3, tickcolor="lightgray"),
        yaxis=dict(domain=HY, range=[y_min, y_max], title=Y_LABEL,
                   fixedrange=True, showgrid=False, showline=True, mirror=True,
                   linecolor="lightgray", linewidth=1, ticks="outside", ticklen=3, tickcolor="lightgray"),

        xaxis2=dict(domain=PX, range=[profile_xmin, profile_xmax],
                    title="Excitation/Emission wavelength [nm]",
                    anchor="y2",
                    showgrid=True, gridcolor="rgba(0,0,0,0.5)", gridwidth=1, zeroline=False,
                    tickmode="linear", tick0=50.0 * floor(profile_xmin / 50.0), dtick=50.0,
                    showline=True, mirror=True, linecolor="lightgray", linewidth=1,
                    ticks="outside", ticklen=3, tickcolor="lightgray",
                    fixedrange=True),
        yaxis2=dict(domain=PY, range=[0, 1],
                    title=Z_LABEL,
                    anchor="x2",
                    showgrid=True, gridcolor="rgba(0,0,0,0.5)", gridwidth=1, zeroline=False,
                    tickmode="linear", tick0=0.0, dtick=0.2, tickformat=".1f",
                    showline=True, mirror=True, linecolor="lightgray", linewidth=1,
                    ticks="outside", ticklen=3, tickcolor="lightgray",
                    fixedrange=True),

        width=FIG_W, height=FIG_H,
        hovermode="closest", clickmode="event+select",
        paper_bgcolor="white", plot_bgcolor="white",
        showlegend=False,
        meta=dict(
            X=x_vals.astype(float).tolist(),
            Y=y_vals.astype(float).tolist(),
            Z=[[None if np.isnan(v) else float(v) for v in row] for row in Z_plot],
            TITLE_X=float(title_x), TITLE_Y=float(title_y)
        )
    )

    # On-click interactivity (crosshair + profiles + profile title; keep yaxis2 locked to [0,1])
    post_js = r"""
    (function(){
      function gd(){var d=document.getElementsByClassName('plotly-graph-div');return d[d.length-1];}
      function ready(cb){var n=0;(function t(){var g=gd();if(window.Plotly&&g&&g.data&&g._fullLayout) return cb(g);
        if(n++<240) setTimeout(t,25);})(); }
      function iNear(a,v){var i=0,m=1/0;for(var k=0;k<a.length;k++){var d=Math.abs(a[k]-v);if(d<m){m=d;i=k;}}return i;}
      function norm(arr){var t=arr.filter(v=>v!=null&&isFinite(v));var m=t.length?Math.max.apply(null,t):0;
        return arr.map(v=>(v==null||!isFinite(v))?null:(m>0?v/m:0));}
      function getTitleIdx(g){
        var A=g.layout.annotations||[], M=g._fullLayout.meta||{};
        var tx=M.TITLE_X, ty=M.TITLE_Y;
        for(var i=0;i<A.length;i++){
          if(A[i].xref==='paper' && A[i].yref==='paper' && Math.abs(A[i].x-tx)<1e-6 && Math.abs(A[i].y-ty)<1e-6) return i;
        }
        return -1;
      }
      ready(function(g){
        var M=g._fullLayout.meta||{}, X=M.X||[], Y=M.Y||[], Z=M.Z||[];
        var iHeat=g._fullData.findIndex(t=>t.uid==='heat');
        var iEm  =g._fullData.findIndex(t=>t.uid==='prof_em');
        var iEx  =g._fullData.findIndex(t=>t.uid==='prof_ex');
        var iTitle=getTitleIdx(g);

        g.on('plotly_click', function(ev){
          var pt=(ev&&ev.points||[]).find(p=>p.curveNumber===iHeat); if(!pt) return;
          var ix=iNear(X, pt.x);
          var iy=iNear(Y, pt.y);
          // Move crosshair
          Plotly.relayout(g,{
            'shapes[0].y0':Y[iy],'shapes[0].y1':Y[iy],
            'shapes[1].x0':X[ix],'shapes[1].x1':X[ix]
          });
          // Update profiles
          var row=Z[iy]||[], col=Z.map(r=>r?r[ix]:null);
          Plotly.restyle(g,{x:[X],y:[norm(row)]},[iEm]);
          Plotly.restyle(g,{x:[Y],y:[norm(col)]},[iEx]);
          // Update profile title
          if(iTitle>=0){
            var txt='Emission@'+Math.round(X[ix])+' nm | Excitation@'+Math.round(Y[iy])+' nm';
            var up={}; up['annotations['+iTitle+'].text']=txt; Plotly.relayout(g, up);
          }
          // Keep y-axis of profile locked to [0,1]
          Plotly.relayout(g, {'yaxis2.range':[0,1]});
        });
      });
    })();
    """

    # Write HTML (no automatic preview)
    pio.write_html(fig, file=out_path, include_plotlyjs="cdn",
                   post_script=post_js, config={"displayModeBar": False})
    print(f"Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="EEM interactive viewer (.csv → .html)")
    p.add_argument("csv", help="Path to input CSV (cols: Em[nm], Ex[nm], Intensity)")
    p.add_argument("--rayleigh-margin", type=float, default=0.0,
                   help="Mask width around Rayleigh line in nm (default: 0.0)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render_one(args.csv, rayleigh_margin_nm=args.rayleigh_margin)
