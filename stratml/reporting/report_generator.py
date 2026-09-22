"""
reporting/report_generator.py
------------------------------
Public API for the reporting module.
Coordinates pdf_builder and comparison — no logic lives here.

Public functions (unchanged interface):
    generate_report(run_id, dataset_name, output_dir) -> Path
    generate_model_script(run_id, output_dir, records) -> Path
"""

from __future__ import annotations

import json
from pathlib import Path

from stratml.reporting.pdf_builder import build_pdf
from stratml.reporting.comparison import write_comparison, generate_model_script  # noqa: F401 — re-exported


def generate_report(run_id: str, dataset_name: str, output_dir: Path) -> Path:
    """Build PDF + comparison files from decision logs. Returns PDF path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir   = output_dir / "decision_logs"
    log_files = sorted(log_dir.glob(f"{run_id}_*.json")) if log_dir.exists() else []

    records = []
    for lf in log_files:
        try:
            records.append(json.loads(lf.read_text()))
        except Exception:
            continue

    write_comparison(records, output_dir)

    pdf_path = output_dir / "report.pdf"
    build_pdf(run_id, dataset_name, records, output_dir, pdf_path)
    return pdf_path


# ---------------------------------------------------------------------------

def _write_comparison(records: list[dict], output_dir: Path) -> None:
    """Write comparison.csv and comparison.json side by side."""
    rows = []
    for rec in records:
        m   = rec["state_snapshot"]["metrics"]
        sec = m.get("secondary", {})
        sig = rec["state_snapshot"]["signals"]
        active_signals = [k for k, v in sig.items()
                          if not k.endswith("_confidence") and v != "none"]
        rows.append({
            "iteration":          rec.get("iteration"),
            "model":              rec["state_snapshot"]["model"]["model_name"],
            "action":             rec["selected_action"]["action_type"],
            "trigger":            rec["selected_action"]["reason"]["trigger"],
            "confidence":         rec["selected_action"].get("confidence"),
            "primary_metric":     m.get("primary"),
            "train_val_gap":      m.get("train_val_gap"),
            "accuracy":           sec.get("accuracy"),
            "f1_score":           sec.get("f1_score"),
            "precision":          sec.get("precision"),
            "recall":             sec.get("recall"),
            "mse":                sec.get("mse"),
            "r2":                 sec.get("r2"),
            "slope":              rec["state_snapshot"]["trajectory"]["slope"],
            "volatility":         rec["state_snapshot"]["trajectory"]["volatility"],
            "runtime":            rec["state_snapshot"]["resources"]["runtime"],
            "active_signals":     ", ".join(active_signals),
        })

    if not rows:
        return

    csv_path = output_dir / "comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "comparison.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _build_pdf(
    run_id: str,
    dataset_name: str,
    records: list[dict],
    output_dir: Path,
    pdf_path: Path,
) -> None:
    doc    = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    h1  = ParagraphStyle("h1",  parent=styles["Title"],   fontSize=18, spaceAfter=4)
    h2  = ParagraphStyle("h2",  parent=styles["Heading2"],fontSize=12, spaceBefore=12, spaceAfter=4)
    bod = styles["BodyText"]
    mon = ParagraphStyle("mon", parent=bod, fontName="Courier", fontSize=8)

    story = []

    # Header
    story.append(Paragraph("StratML — Execution Report", h1))
    story.append(Paragraph(f"Run ID: <b>{run_id}</b>  |  Dataset: <b>{dataset_name}</b>", bod))
    story.append(Paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", bod))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey, spaceAfter=8))

    # Summary
    story.append(Paragraph("Summary & Computational Budget", h2))
    story.append(Paragraph(f"Iterations logged: <b>{len(records)}</b>", bod))
    story.append(Paragraph(f"Output directory: <font name='Courier'>{output_dir}</font>", bod))
    budget_file = output_dir / "artifacts" / "budget_accounting.json"
    if budget_file.exists():
        try:
            bdata = json.loads(budget_file.read_text(encoding="utf-8"))
            act = bdata.get("actual_consumption", {})
            cfg = bdata.get("configured_budget", {})
            comp = bdata.get("paper_compliance", {})
            story.append(Paragraph(f"Model Fits / Evaluations: <b>{act.get('model_fits', len(records))}</b>", bod))
            story.append(Paragraph(f"Total Runtime: <b>{act.get('runtime_seconds', 0):.2f}s</b> (Timeout: {cfg.get('timeout_per_run_seconds')}s, soft boundary)", bod))
            is_paper = comp.get("is_paper_standard", True)
            b_label = "Paper Standard (tune=False)" if is_paper else "Exploratory Tuned (tune=True - NON-PAPER)"
            story.append(Paragraph(f"Budget Mode: <b>{b_label}</b>", bod))
        except Exception:
            pass

    # Iteration comparison table
    story.append(Paragraph("Iteration Comparison", h2))
    tdata = [["Iter", "Model", "Action", "Trigger", "Conf", "Primary", "Gap", "Signals"]]
    best_metric = 0.0
    best_model  = "—"
    best_iter   = 0

    for rec in records:
        it     = rec.get("iteration", "?")
        m      = rec["state_snapshot"]["metrics"]
        sig    = rec["state_snapshot"]["signals"]
        active = [k for k, v in sig.items() if not k.endswith("_confidence") and v != "none"]
        primary = m.get("primary", 0.0) or 0.0
        tdata.append([
            str(it),
            rec["state_snapshot"]["model"]["model_name"],
            rec["selected_action"]["action_type"],
            rec["selected_action"]["reason"]["trigger"],
            f"{rec['selected_action'].get('confidence') or 0:.2f}",
            f"{primary:.4f}",
            f"{m.get('train_val_gap') or 0:.4f}",
            ", ".join(active) or "—",
        ])
        if primary > best_metric:
            best_metric = primary
            best_model  = rec["state_snapshot"]["model"]["model_name"]
            best_iter   = it

    cw = [1*cm, 4*cm, 3.5*cm, 3*cm, 1.5*cm, 2.2*cm, 2*cm, 4*cm]
    t  = Table(tdata, colWidths=cw, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0),  colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",      (0,0), (-1,0),  colors.white),
        ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,-1), 7),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("GRID",           (0,0), (-1,-1), 0.3, colors.grey),
        ("VALIGN",         (0,0), (-1,-1), "TOP"),
    ]))
    story.append(t)

    # Model comparison (unique models)
    story.append(Paragraph("Model Comparison", h2))
    model_map: dict[str, dict] = {}
    for rec in records:
        mn  = rec["state_snapshot"]["model"]["model_name"]
        pri = rec["state_snapshot"]["metrics"].get("primary") or 0.0
        sec = rec["state_snapshot"]["metrics"].get("secondary", {})
        if mn not in model_map or pri > model_map[mn]["primary"]:
            model_map[mn] = {"primary": pri, "f1": sec.get("f1_score"), "acc": sec.get("accuracy")}

    mdata = [["Model", "Best Primary", "Accuracy", "F1 Score"]]
    for mn, v in model_map.items():
        mdata.append([mn, f"{v['primary']:.4f}",
                       f"{v['acc']:.4f}" if v["acc"] is not None else "—",
                       f"{v['f1']:.4f}"  if v["f1"]  is not None else "—"])
    mt = Table(mdata, colWidths=[5*cm, 3.5*cm, 3.5*cm, 3.5*cm], repeatRows=1)
    mt.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0),  colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",      (0,0), (-1,0),  colors.white),
        ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("GRID",           (0,0), (-1,-1), 0.3, colors.grey),
    ]))
    story.append(mt)

    # Best model
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Best Model", h2))
    story.append(Paragraph(f"Model: <b>{best_model}</b>  |  Iteration: <b>{best_iter}</b>  |  Primary metric: <b>{best_metric:.4f}</b>", bod))
    model_file = output_dir / "artifacts" / "model.pkl"
    if model_file.exists():
        story.append(Paragraph(f"Saved at: <font name='Courier'>{model_file}</font>", mon))

    # Final metrics from artifact
    metrics_file = output_dir / "artifacts" / "metrics.json"
    if metrics_file.exists():
        story.append(Paragraph("Final Validation Metrics", h2))
        for k, v in json.loads(metrics_file.read_text()).items():
            if v is not None:
                story.append(Paragraph(f"<font name='Courier'>{k}: {v}</font>", mon))

    # Test metrics
    test_metrics_file = output_dir / "artifacts" / "test_metrics.json"
    if test_metrics_file.exists():
        story.append(Paragraph("Test Set Metrics", h2))
        story.append(Paragraph(
            "Evaluated on the held-out test split — these metrics were not seen during training.",
            bod,
        ))
        for k, v in json.loads(test_metrics_file.read_text()).items():
            if v is not None:
                story.append(Paragraph(f"<font name='Courier'>{k}: {v}</font>", mon))

    doc.build(story)


def generate_model_script(
    run_id: str,
    output_dir: Path,
    records: list[dict],
) -> Path:
    """Generate a standalone model.py that loads and uses the best model."""
    best_metric = -1.0
    best_model_name = "UnknownModel"
    best_params: dict = {}

    for rec in records:
        primary = rec["state_snapshot"]["metrics"].get("primary") or 0.0
        if primary > best_metric:
            best_metric     = primary
            best_model_name = rec["state_snapshot"]["model"]["model_name"]
            best_params     = rec["state_snapshot"]["model"].get("hyperparameters", {})

    model_pkl = output_dir / "artifacts" / "model.pkl"
    script = f'''"""
model.py — Auto-generated by StratML
Run ID  : {run_id}
Model   : {best_model_name}
Metric  : {best_metric:.4f}
"""

import joblib
import numpy as np
from pathlib import Path

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(r"{model_pkl}")
model = joblib.load(MODEL_PATH)

# ── Hyperparameters used ──────────────────────────────────────────────────────
hyperparameters = {best_params!r}

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(X):
    """
    X: array-like of shape (n_samples, n_features)
    Returns: predicted labels
    """
    return model.predict(np.array(X))


def predict_proba(X):
    """Returns class probabilities if the model supports it."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(np.array(X))
    raise NotImplementedError(f"{{type(model).__name__}} does not support predict_proba")


if __name__ == "__main__":
    print(f"Model : {{type(model).__name__}}")
    print(f"Params: {{model.get_params()}}")
    print("Ready. Call predict(X) or predict_proba(X).")
'''

    script_path = output_dir / "artifacts" / "model.py"
    script_path.write_text(script, encoding="utf-8")
    return script_path
