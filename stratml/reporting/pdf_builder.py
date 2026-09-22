"""
reporting/pdf_builder.py
------------------------
PDF generation using ReportLab — professional A4 report with charts.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1a2744")
BLUE   = colors.HexColor("#2563eb")
LIGHT  = colors.HexColor("#f0f4ff")
ALT    = colors.HexColor("#f8fafc")
BORDER = colors.HexColor("#cbd5e1")
GREEN  = colors.HexColor("#16a34a")
RED    = colors.HexColor("#dc2626")
AMBER  = colors.HexColor("#d97706")
SLATE  = colors.HexColor("#64748b")

# Generous cell padding so text never touches borders
_PAD = 7


def _ts():
    s = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("_t",  parent=s["Title"],   fontSize=22, textColor=NAVY, spaceAfter=2, leading=28),
        "sub":   ParagraphStyle("_s",  parent=s["Normal"],  fontSize=14, textColor=BLUE, spaceAfter=6),
        "h2":    ParagraphStyle("_h2", parent=s["Heading2"],fontSize=11, textColor=NAVY, spaceBefore=18, spaceAfter=6),
        "body":  ParagraphStyle("_b",  parent=s["Normal"],  fontSize=9,  textColor=colors.HexColor("#334155"), leading=14, spaceAfter=3),
        "cell":  ParagraphStyle("_c",  parent=s["Normal"],  fontSize=8,  textColor=colors.HexColor("#1e293b"), leading=11),
        "mono":  ParagraphStyle("_m",  parent=s["Normal"],  fontName="Courier", fontSize=7.5, textColor=NAVY, leading=11),
    }


def _base_style(header_cols=None):
    """Base TableStyle applied to every table."""
    cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, ALT]),
        ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), _PAD),
        ("BOTTOMPADDING", (0, 0), (-1, -1), _PAD),
        ("LEFTPADDING",   (0, 0), (-1, -1), _PAD),
        ("RIGHTPADDING",  (0, 0), (-1, -1), _PAD),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    return TableStyle(cmds)


def _section(story, title, st):
    story.append(Paragraph(title, st["h2"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=6))


# ── Section builders ──────────────────────────────────────────────────────────

def _build_header(run_id, dataset_name, n_iters):
    t = Table(
        [["Run ID",    run_id,    "Dataset",    dataset_name],
         ["Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
          "Iterations", str(n_iters)]],
        colWidths=[2.5*cm, 7*cm, 2.5*cm, 3.5*cm],
    )
    t.setStyle(TableStyle([
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR",     (0, 0), (0, -1), SLATE),
        ("TEXTCOLOR",     (2, 0), (2, -1), SLATE),
        ("TEXTCOLOR",     (1, 0), (1, -1), NAVY),
        ("TEXTCOLOR",     (3, 0), (3, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _build_kpi(best, cell_s):
    sec   = best["state_snapshot"]["metrics"].get("secondary", {})
    pri   = best["state_snapshot"]["metrics"].get("primary", 0) or 0
    gap   = best["state_snapshot"]["metrics"].get("train_val_gap", 0) or 0
    f1v   = sec.get("f1_score")
    model = best["state_snapshot"]["model"]["model_name"]
    gap_color = "#16a34a" if gap < 0.05 else ("#d97706" if gap < 0.15 else "#dc2626")

    t = Table([[
        Paragraph('<font color="#64748b" size="7">BEST MODEL</font><br/>'
                  f'<font color="#1a2744" size="10"><b>{model}</b></font>', cell_s),
        Paragraph('<font color="#64748b" size="7">PRIMARY METRIC</font><br/>'
                  f'<font color="#16a34a" size="14"><b>{pri:.4f}</b></font>', cell_s),
        Paragraph('<font color="#64748b" size="7">TRAIN / VAL GAP</font><br/>'
                  f'<font color="{gap_color}" size="12"><b>{gap:.4f}</b></font>', cell_s),
        Paragraph('<font color="#64748b" size="7">F1 SCORE</font><br/>'
                  f'<font color="#1a2744" size="10"><b>{f"{f1v:.4f}" if f1v else "—"}</b></font>', cell_s),
    ]], colWidths=[4.2*cm, 4.2*cm, 4.2*cm, 4.2*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT),
        ("BOX",           (0, 0), (0, -1),  0.5, BORDER),
        ("BOX",           (1, 0), (1, -1),  0.5, BORDER),
        ("BOX",           (2, 0), (2, -1),  0.5, BORDER),
        ("BOX",           (3, 0), (3, -1),  0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _build_budget_card(output_dir: Path, records: list[dict], cell_s):
    import json
    budget_file = output_dir / "artifacts" / "budget_accounting.json"
    bdata = None
    if budget_file.exists():
        try:
            bdata = json.loads(budget_file.read_text(encoding="utf-8"))
        except Exception:
            bdata = None

    if bdata:
        cfg = bdata.get("configured_budget", {})
        act = bdata.get("actual_consumption", {})
        comp = bdata.get("paper_compliance", {})
        iters = act.get("decision_iterations", len(records))
        fits = act.get("model_fits", len(records))
        runtime = act.get("runtime_seconds", sum(r.get("state_snapshot", {}).get("resources", {}).get("runtime", 0) for r in records))
        is_paper = comp.get("is_paper_standard", True)
        timeout_val = cfg.get("timeout_per_run_seconds")
    else:
        iters = len(records)
        fits = len(records)
        runtime = sum(r.get("state_snapshot", {}).get("resources", {}).get("runtime", 0) for r in records)
        is_paper = True
        timeout_val = 300

    profile_label = "Paper Standard (tune=False)" if is_paper else "Exploratory (tune=True)"
    profile_color = "#16a34a" if is_paper else "#d97706"

    t = Table([[
        Paragraph('<font color="#64748b" size="7">DECISION ITERATIONS</font><br/>'
                  f'<font color="#1a2744" size="12"><b>{iters}</b></font>', cell_s),
        Paragraph('<font color="#64748b" size="7">MODEL FITS / EVALS</font><br/>'
                  f'<font color="#1a2744" size="12"><b>{fits}</b></font>', cell_s),
        Paragraph('<font color="#64748b" size="7">TOTAL RUNTIME</font><br/>'
                  f'<font color="#1a2744" size="12"><b>{runtime:.2f}s</b></font><br/>'
                  f'<font color="#64748b" size="6.5">Timeout: {timeout_val}s (soft)</font>', cell_s),
        Paragraph('<font color="#64748b" size="7">BUDGET PROFILE</font><br/>'
                  f'<font color="{profile_color}" size="8.5"><b>{profile_label}</b></font>', cell_s),
    ]], colWidths=[4.2*cm, 4.2*cm, 4.2*cm, 4.2*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), ALT),
        ("BOX",           (0, 0), (0, -1),  0.5, BORDER),
        ("BOX",           (1, 0), (1, -1),  0.5, BORDER),
        ("BOX",           (2, 0), (2, -1),  0.5, BORDER),
        ("BOX",           (3, 0), (3, -1),  0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t, bdata


def _build_dataset(ds):
    """Dataset profile table from state_snapshot.dataset."""
    cd = ds.get("class_distribution", {})
    dist_str = "  |  ".join(f"{k}: {v}" for k, v in cd.items()) if cd else "—"
    rows = [
        ["Samples",          f'{ds.get("num_samples", "—"):,}' if isinstance(ds.get("num_samples"), int) else "—",
         "Features",         str(ds.get("num_features", "—"))],
        ["Missing ratio",    f'{ds.get("missing_ratio", 0):.1%}',
         "Imbalance ratio",  str(ds.get("imbalance_ratio", "—"))],
        ["Class distribution", dist_str, "", ""],
    ]
    t = Table(rows, colWidths=[3.5*cm, 4*cm, 3.5*cm, 4.8*cm])
    ts = TableStyle([
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR",     (0, 0), (0, -1), SLATE),
        ("TEXTCOLOR",     (2, 0), (2, -1), SLATE),
        ("TEXTCOLOR",     (1, 0), (1, -1), NAVY),
        ("TEXTCOLOR",     (3, 0), (3, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("SPAN",          (1, 2), (3, 2)),
        ("BACKGROUND",    (0, 0), (-1, -1), ALT),
        ("BOX",           (0, 0), (-1, -1), 0.4, BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, BORDER),
    ])
    t.setStyle(ts)
    return t


def _build_metrics_table(records):
    """Full metrics per iteration: Primary, Accuracy, Precision, Recall, F1, MSE/R2, Gap."""
    hdr = ["#", "Model", "Primary", "Accuracy", "Precision", "Recall", "F1", "MSE / R²", "Gap"]
    rows = [hdr]
    for rec in records:
        m   = rec["state_snapshot"]["metrics"]
        sec = m.get("secondary", {})
        pri = m.get("primary", 0) or 0
        gap = m.get("train_val_gap", 0) or 0
        acc = sec.get("accuracy")
        pre = sec.get("precision")
        rec_ = sec.get("recall")
        f1  = sec.get("f1_score")
        mse = sec.get("mse")
        r2  = sec.get("r2")
        mse_r2 = f"{mse:.4f}" if mse is not None else (f"{r2:.4f}" if r2 is not None else "—")
        rows.append([
            str(rec.get("iteration", "?")),
            rec["state_snapshot"]["model"]["model_name"][:22],
            f"{pri:.4f}",
            f"{acc:.4f}" if acc is not None else "—",
            f"{pre:.4f}" if pre is not None else "—",
            f"{rec_:.4f}" if rec_ is not None else "—",
            f"{f1:.4f}"  if f1  is not None else "—",
            mse_r2,
            f"{gap:.4f}",
        ])

    ts = _base_style()
    ts.add("ALIGN", (0, 0), (0, -1), "CENTER")
    ts.add("ALIGN", (2, 0), (-1, -1), "RIGHT")
    for i, rec in enumerate(records, start=1):
        gap = rec["state_snapshot"]["metrics"].get("train_val_gap", 0) or 0
        gc = GREEN if gap < 0.05 else (AMBER if gap < 0.15 else RED)
        ts.add("TEXTCOLOR", (8, i), (8, i), gc)
        ts.add("FONTNAME",  (8, i), (8, i), "Helvetica-Bold")

    t = Table(rows,
              colWidths=[0.7*cm, 3.8*cm, 1.9*cm, 2*cm, 2*cm, 1.9*cm, 1.9*cm, 2*cm, 1.8*cm],
              repeatRows=1)
    t.setStyle(ts)
    return t


def _build_hyperparams_table(records, mono_s):
    """One row per iteration showing the hyperparameters used."""
    hdr = ["#", "Model", "Hyperparameters"]
    rows = [hdr]
    for rec in records:
        hp = rec["state_snapshot"]["model"].get("hyperparameters") or {}
        hp_str = "  |  ".join(f"{k}={v}" for k, v in hp.items()) if hp else "defaults"
        rows.append([
            str(rec.get("iteration", "?")),
            rec["state_snapshot"]["model"]["model_name"][:22],
            Paragraph(hp_str, mono_s),
        ])

    ts = _base_style()
    ts.add("ALIGN", (0, 0), (0, -1), "CENTER")
    ts.add("VALIGN", (2, 1), (2, -1), "TOP")

    t = Table(rows, colWidths=[0.7*cm, 3.8*cm, 12.3*cm], repeatRows=1)
    t.setStyle(ts)
    return t


def _build_agent_table(records):
    """Agent scores + candidates considered per iteration."""
    hdr = ["#", "Action Taken", "Perf", "Eff", "Stab", "Candidates Considered", "Trigger", "Conf"]
    rows = [hdr]
    for rec in records:
        sel   = rec["selected_action"]
        ag    = sel.get("agent_scores", {})
        cands = rec.get("candidate_actions", [])
        cand_str = ", ".join(c["action_type"] for c in cands) if cands else "—"
        conf  = sel.get("confidence", 0) or 0
        rows.append([
            str(rec.get("iteration", "?")),
            sel["action_type"],
            f'{ag.get("performance", 0):.2f}' if ag.get("performance") is not None else "—",
            f'{ag.get("efficiency",  0):.2f}' if ag.get("efficiency")  is not None else "—",
            f'{ag.get("stability",   0):.2f}' if ag.get("stability")   is not None else "—",
            cand_str,
            sel["reason"]["trigger"],
            f"{conf:.2f}",
        ])

    ts = _base_style()
    ts.add("ALIGN", (0, 0), (0, -1), "CENTER")
    ts.add("ALIGN", (2, 0), (4, -1), "CENTER")
    ts.add("ALIGN", (7, 0), (7, -1), "CENTER")
    # Colour confidence
    for i, rec in enumerate(records, start=1):
        conf = rec["selected_action"].get("confidence", 0) or 0
        cc = GREEN if conf >= 0.7 else (AMBER if conf >= 0.5 else RED)
        ts.add("TEXTCOLOR", (7, i), (7, i), cc)
        ts.add("FONTNAME",  (7, i), (7, i), "Helvetica-Bold")

    t = Table(rows,
              colWidths=[0.7*cm, 3.5*cm, 1.5*cm, 1.5*cm, 1.5*cm, 4.5*cm, 2.5*cm, 1.6*cm],
              repeatRows=1)
    t.setStyle(ts)
    return t


def _build_trace(records, body_s):
    """Narrative decision trace paragraphs."""
    items = []
    for rec in records:
        sel  = rec["selected_action"]
        conf = sel.get("confidence", 0) or 0
        cc   = "#16a34a" if conf >= 0.7 else ("#d97706" if conf >= 0.5 else "#dc2626")
        sig  = rec["state_snapshot"]["signals"]
        active = ", ".join(k for k, v in sig.items()
                           if not k.endswith("_confidence") and v != "none")
        ev   = sel.get("reason", {}).get("evidence", {})
        ev_parts = [f"{k}={v}" for k, v in ev.items()
                    if k not in ("trigger", "confidence") and v is not None]

        line = (
            f'<font color="#64748b">Iter {rec.get("iteration","?")}  </font>'
            f'<b>{rec["state_snapshot"]["model"]["model_name"]}</b>'
            f'<font color="#64748b">  →  </font>'
            f'<font color="#2563eb"><b>{sel["action_type"]}</b></font>'
            f'<font color="#64748b">  trigger=</font>{sel["reason"]["trigger"]}'
            f'<font color="#64748b">  conf=</font><font color="{cc}"><b>{conf:.2f}</b></font>'
        )
        if active:
            line += f'<font color="#64748b">  signals=</font><i>{active}</i>'
        if ev_parts:
            line += f'<font color="#64748b">  evidence=</font><i>{",  ".join(ev_parts)}</i>'
        items.append(Paragraph(line, body_s))
    return items


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _fig_to_image(fig, width_cm: float, height_cm: float) -> Image:
    """Render a matplotlib figure to an in-memory PNG and return a ReportLab Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return Image(buf, width=width_cm * cm, height=height_cm * cm)


def _chart_performance(records: list[dict]) -> Image | None:
    """Line chart: primary metric per iteration."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        iters   = [r["iteration"] for r in records]
        primary = [r["state_snapshot"]["metrics"].get("primary") or 0 for r in records]
        gaps    = [r["state_snapshot"]["metrics"].get("train_val_gap") or 0 for r in records]
        models  = [r["state_snapshot"]["model"]["model_name"] for r in records]

        fig, ax1 = plt.subplots(figsize=(7.5, 2.8))
        fig.patch.set_facecolor("#f8fafc")
        ax1.set_facecolor("#f8fafc")

        ax1.plot(iters, primary, "o-", color="#2563eb", linewidth=2, markersize=6,
                 label="Primary metric", zorder=3)
        ax1.fill_between(iters, primary, alpha=0.08, color="#2563eb")

        ax2 = ax1.twinx()
        ax2.bar(iters, gaps, alpha=0.25, color="#dc2626", width=0.4, label="Train/val gap")
        ax2.set_ylabel("Train/val gap", fontsize=7, color="#dc2626")
        ax2.tick_params(axis="y", labelcolor="#dc2626", labelsize=7)
        ax2.set_ylim(0, max(max(gaps) * 2.5, 0.3))

        # Annotate model names
        for i, (x, y, m) in enumerate(zip(iters, primary, models)):
            short = m.replace("Classifier", "").replace("Regressor", "")[:14]
            ax1.annotate(short, (x, y), textcoords="offset points", xytext=(0, 7),
                         ha="center", fontsize=6, color="#1a2744")

        ax1.set_xlabel("Iteration", fontsize=8)
        ax1.set_ylabel("Primary metric", fontsize=8, color="#2563eb")
        ax1.tick_params(axis="y", labelcolor="#2563eb", labelsize=7)
        ax1.tick_params(axis="x", labelsize=7)
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax1.set_title("Performance & Generalization Gap per Iteration", fontsize=9,
                      color="#1a2744", pad=6)
        ax1.grid(axis="y", linestyle="--", alpha=0.4, color="#cbd5e1")
        ax1.spines[["top", "right"]].set_visible(False)
        ax2.spines[["top"]].set_visible(False)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower right",
                   framealpha=0.7)
        fig.tight_layout()
        return _fig_to_image(fig, 16.6, 6.5)
    except Exception:
        return None


def _chart_signals(records: list[dict]) -> Image | None:
    """Heatmap: signal strength per iteration."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        _SIGNAL_KEYS = ["underfitting", "overfitting", "well_fitted", "converged",
                        "stagnating", "too_slow", "diminishing_returns", "diverging"]
        _STRENGTH = {"none": 0, "weak": 1, "strong": 2}

        iters = [r["iteration"] for r in records]
        matrix = []
        for key in _SIGNAL_KEYS:
            row = [_STRENGTH.get(r["state_snapshot"]["signals"].get(key, "none"), 0)
                   for r in records]
            matrix.append(row)

        data = np.array(matrix, dtype=float)
        # Skip all-zero rows
        active_mask = data.any(axis=1)
        if not active_mask.any():
            return None
        data    = data[active_mask]
        labels  = [k for k, m in zip(_SIGNAL_KEYS, active_mask) if m]

        fig, ax = plt.subplots(figsize=(7.5, max(1.8, len(labels) * 0.55 + 0.6)))
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")

        cmap = plt.cm.get_cmap("RdYlGn_r", 3)
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=2,
                       interpolation="nearest")

        ax.set_xticks(range(len(iters)))
        ax.set_xticklabels([f"Iter {i}" for i in iters], fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title("Signal Heatmap (0=none, 1=weak, 2=strong)", fontsize=9,
                     color="#1a2744", pad=6)

        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], fraction=0.03, pad=0.02)
        cbar.ax.set_yticklabels(["none", "weak", "strong"], fontsize=6)

        # Cell text
        for r in range(len(labels)):
            for c in range(len(iters)):
                v = int(data[r, c])
                if v > 0:
                    ax.text(c, r, ["", "W", "S"][v], ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")

        fig.tight_layout()
        return _fig_to_image(fig, 16.6, max(3.5, len(labels) * 0.7 + 1.2))
    except Exception:
        return None


def _chart_runtime(records: list[dict]) -> Image | None:
    """Horizontal bar chart: runtime per model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models   = [r["state_snapshot"]["model"]["model_name"] for r in records]
        runtimes = [r["state_snapshot"]["model"].get("runtime") or 0 for r in records]
        iters    = [r["iteration"] for r in records]

        if all(rt == 0 for rt in runtimes):
            return None

        labels = [f"Iter {i}: {m[:18]}" for i, m in zip(iters, models)]
        colors_bar = ["#2563eb" if rt < 60 else "#d97706" if rt < 300 else "#dc2626"
                      for rt in runtimes]

        fig, ax = plt.subplots(figsize=(7.5, max(2.0, len(records) * 0.45 + 0.8)))
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")

        bars = ax.barh(labels, runtimes, color=colors_bar, height=0.55, edgecolor="white")
        for bar, rt in zip(bars, runtimes):
            ax.text(bar.get_width() + max(runtimes) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{rt:.1f}s", va="center", fontsize=7, color="#334155")

        ax.set_xlabel("Runtime (seconds)", fontsize=8)
        ax.set_title("Training Runtime per Iteration", fontsize=9, color="#1a2744", pad=6)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.4, color="#cbd5e1")
        ax.invert_yaxis()
        fig.tight_layout()
        return _fig_to_image(fig, 16.6, max(3.0, len(records) * 0.6 + 1.2))
    except Exception:
        return None


def _chart_agent_scores(records: list[dict]) -> Image | None:
    """Grouped bar chart: agent scores per iteration."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        iters = [r["iteration"] for r in records]
        perf  = [r["selected_action"].get("agent_scores", {}).get("performance") or 0 for r in records]
        eff   = [r["selected_action"].get("agent_scores", {}).get("efficiency")  or 0 for r in records]
        stab  = [r["selected_action"].get("agent_scores", {}).get("stability")   or 0 for r in records]

        if all(v == 0 for v in perf + eff + stab):
            return None

        x = np.arange(len(iters))
        w = 0.25
        fig, ax = plt.subplots(figsize=(7.5, 2.8))
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")

        ax.bar(x - w, perf, w, label="Performance", color="#2563eb", alpha=0.85)
        ax.bar(x,     eff,  w, label="Efficiency",  color="#16a34a", alpha=0.85)
        ax.bar(x + w, stab, w, label="Stability",   color="#d97706", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Iter {i}" for i in iters], fontsize=7)
        ax.set_ylabel("Score", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title("Agent Scores for Selected Action", fontsize=9, color="#1a2744", pad=6)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4, color="#cbd5e1")
        fig.tight_layout()
        return _fig_to_image(fig, 16.6, 5.5)
    except Exception:
        return None


# ── Public entry point ────────────────────────────────────────────────────────

def build_pdf(
    run_id: str,
    dataset_name: str,
    records: list[dict],
    output_dir: Path,
    pdf_path: Path,
) -> None:
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2*cm,
    )
    st    = _ts()
    story = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("StratML", st["title"]))
    story.append(Paragraph("Experiment Report", st["sub"]))
    story.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=8))
    story.append(_build_header(run_id, dataset_name, len(records)))
    story.append(Spacer(1, 0.35*cm))

    # ── Computational Budget ──────────────────────────────────────────────────
    budget_card, bdata = _build_budget_card(output_dir, records, st["cell"])
    story.append(budget_card)
    story.append(Spacer(1, 0.35*cm))

    if bdata and not bdata.get("paper_compliance", {}).get("is_paper_standard", True):
        story.append(Paragraph(
            "<font color='#d97706'><b>Budget Boundary Note:</b></font> "
            "This experiment executed with hyperparameter tuning enabled (<b>--tune</b>), "
            "which evaluates multiple candidates per iteration via cross-validation. "
            "The target paper budget configuration is <b>max_iterations: 5, tune: false</b>.",
            st["cell"]
        ))
        story.append(Spacer(1, 0.3*cm))

    # ── KPI cards ─────────────────────────────────────────────────────────────
    if records:
        best = max(records, key=lambda r: r["state_snapshot"]["metrics"].get("primary") or 0)
        story.append(_build_kpi(best, st["cell"]))
        story.append(Spacer(1, 0.4*cm))

    # ── Dataset profile ───────────────────────────────────────────────────────
    ds = (records[0]["state_snapshot"].get("dataset") or {}) if records else {}
    if ds:
        _section(story, "Dataset Profile", st)
        story.append(_build_dataset(ds))
        story.append(Spacer(1, 0.3*cm))

    # ── Full metrics ──────────────────────────────────────────────────────────
    if records:
        _section(story, "Metrics per Iteration", st)
        story.append(_build_metrics_table(records))
        story.append(Spacer(1, 0.3*cm))

    # ── Hyperparameters ───────────────────────────────────────────────────────
    if records:
        _section(story, "Hyperparameters Used", st)
        story.append(_build_hyperparams_table(records, st["mono"]))
        story.append(Spacer(1, 0.3*cm))

    # ── Agent deliberation ────────────────────────────────────────────────────
    if records:
        _section(story, "Agent Deliberation", st)
        story.append(_build_agent_table(records))
        story.append(Spacer(1, 0.3*cm))

    # ── Decision trace ────────────────────────────────────────────────────────
    _section(story, "Decision Trace", st)
    story.extend(_build_trace(records, st["body"]))

    # ── Visualizations ────────────────────────────────────────────────────────
    if records:
        _section(story, "Visualizations", st)

        perf_img = _chart_performance(records)
        if perf_img:
            story.append(perf_img)
            story.append(Spacer(1, 0.4*cm))

        sig_img = _chart_signals(records)
        if sig_img:
            story.append(sig_img)
            story.append(Spacer(1, 0.4*cm))

        agent_img = _chart_agent_scores(records)
        if agent_img:
            story.append(agent_img)
            story.append(Spacer(1, 0.4*cm))

        rt_img = _chart_runtime(records)
        if rt_img:
            story.append(rt_img)

    doc.build(story)
