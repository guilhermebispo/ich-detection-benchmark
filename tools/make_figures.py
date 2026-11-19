"""Figures and summary table of the article, drawn from the result files.

    python tools/make_figures.py DATA_DIR OUT_DIR

DATA_DIR holds the result files: `correcao_retinanet/final_results.json` (corrected RetinaNet, deviation 1),
`final_results.json` (first run, used for the YOLO latencies and for the RetinaNet trained with transposed
boxes) and the cohort counts (`cq500_series_escolhidas.csv`, `export_512_summary.json`,
`ctich_contagens.json`). PDF figures are written to OUT_DIR/figures and the LaTeX table to OUT_DIR/tables.
No number is typed by hand: every value comes from those files.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

_savefig = Figure.savefig
Figure.savefig = lambda self, *a, **k: _savefig(self, *a, metadata={"CreationDate": None}, **k)  # reproducible PDFs
import numpy as np  # noqa: E402

DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2]) / "figures"
CORRECTED = json.loads((DATA / "correcao_retinanet" / "final_results.json").read_text())
FIRST = json.loads((DATA / "final_results.json").read_text())
# Exploratory endpoints (notebook 12, meeting of 5 Nov 2026); not part of the registered plan
EXPLORATORY = json.loads((DATA / "correcao_retinanet" / "exploratory_results.json").read_text())

ARMS = ("retinanet", "yolo_no_mosaic", "yolo_mosaic")
NAME = {"retinanet": "RetinaNet", "yolo_no_mosaic": "YOLOv8m, no mosaic", "yolo_mosaic": "YOLOv8m, mosaic"}
COLOR = {"retinanet": "#2a78d6", "yolo_no_mosaic": "#eb6834", "yolo_mosaic": "#1baf7a"}
MARKER = {"retinanet": "o", "yolo_no_mosaic": "s", "yolo_mosaic": "D"}
GREY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"
MARGIN = 0.05

plt.rcParams.update({
    "pdf.compression": 6,
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42, "savefig.bbox": "tight",
})


def arm_stats(comparison: dict, key: str) -> dict:
    """Statistics of one arm inside a two-arm comparison block (`A` or `B`)."""
    return comparison[key]


def validation(arm: str) -> dict:
    v = CORRECTED["validation"]
    return arm_stats(v["primary_A_vs_B"], "A") if arm == "retinanet" else (
        arm_stats(v["primary_A_vs_B"], "B") if arm == "yolo_no_mosaic" else arm_stats(v["mosaic_C_vs_B"], "A"))


def test(arm: str) -> dict:
    t = CORRECTED["test"]
    return arm_stats(t["A_vs_B"], "A") if arm == "retinanet" else (
        arm_stats(t["A_vs_B"], "B") if arm == "yolo_no_mosaic" else arm_stats(t["C_vs_B"], "A"))


def figure_performance():
    """AP@0.5 of each arm in validation and on the test set, with fold models and the first-run RetinaNet."""
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    offsets = {"retinanet": -0.12, "yolo_no_mosaic": 0.0, "yolo_mosaic": 0.12}
    for arm in ARMS:
        points = [validation(arm), test(arm)]
        x = np.array([0, 1]) + offsets[arm]
        y = [p["ap50"] for p in points]
        ax.plot(x, y, color=COLOR[arm], lw=1.6, zorder=2)
        for xi, p in zip(x, points):
            ax.errorbar(xi, p["ap50"], yerr=[[p["ap50"] - p["ci95"][0]], [p["ci95"][1] - p["ap50"]]],
                        fmt=MARKER[arm], ms=6.5, color=COLOR[arm], mec="white", mew=1.0, elinewidth=1.4,
                        capsize=0, zorder=3)
            ax.scatter(np.full(len(p["per_model_ap50"]), xi + 0.045), p["per_model_ap50"], s=7,
                       color=COLOR[arm], alpha=0.55, lw=0, zorder=2)
        ax.plot([], [], color=COLOR[arm], marker=MARKER[arm], ms=6, mec="white", lw=1.6, label=NAME[arm])
    first = [FIRST["validation"]["primary_A_vs_B"]["A"], FIRST["test"]["A_vs_B"]["A"]]
    x = np.array([0, 1]) - 0.24
    ax.plot(x, [p["ap50"] for p in first], color=GREY, lw=1.2, ls=(0, (3, 2)), zorder=1)
    ax.scatter(x, [p["ap50"] for p in first], marker="o", s=26, facecolor="white", edgecolor=GREY, lw=1.2, zorder=2)
    ax.plot([], [], color=GREY, ls=(0, (3, 2)), marker="o", mfc="white", mec=GREY, label="RetinaNet, first run (transposed boxes)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xticks([0, 1], ["Validation\n(BHX, 5 folds, 436 exams)", "Test\n(CT-ICH, 75 exams)"])
    ax.set_xlim(-0.45, 1.3)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("AP@0.5")
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "results_performance.pdf")
    plt.close(fig)


def figure_differences():
    """Differences in AP@0.5 with 90% CIs against the equivalence margin."""
    v, t = CORRECTED["validation"], CORRECTED["test"]
    rows = [
        ("Validation", "RetinaNet − YOLO, no mosaic (primary)", v["primary_A_vs_B"], True),
        ("Validation", "  pooled out-of-fold predictions", v["sensitivity_pooled"], False),
        ("Validation", "  equivocal exams as negatives", v["sensitivity_equivocal_as_negative"], False),
        ("Validation", "  without BHX/CQ500 disagreement", v["sensitivity_without_bhx_cq500_disagreement"], False),
        ("Validation", "YOLO mosaic − no mosaic", v["mosaic_C_vs_B"], False),
        ("Validation", "RetinaNet − YOLO, mosaic", v["mosaic_A_vs_C"], False),
        ("Test", "RetinaNet − YOLO, no mosaic", t["A_vs_B"], True),
        ("Test", "YOLO mosaic − no mosaic", t["C_vs_B"], False),
        ("Test", "RetinaNet − YOLO, mosaic", t["A_vs_C"], False),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.axvspan(-MARGIN, MARGIN, color="#eef3fb", zorder=0)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    for m in (-MARGIN, MARGIN):
        ax.axvline(m, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    y, ticks, labels, bold, current = 0, [], [], [], None
    for group, label, block, primary in rows:
        if group != current:
            if current is not None:
                y += 0.4
            ticks.append(y)
            labels.append(group)
            bold.append(True)
            y += 1
            current = group
        e = block["equivalence_A_minus_B"]
        lo, hi = e["ci90"]
        ax.plot([lo, hi], [y, y], color=INK, lw=1.6 if primary else 1.1, solid_capstyle="round", zorder=2)
        ax.plot(e["difference"], y, marker="D" if primary else "o", ms=6 if primary else 4.5,
                color=INK, mec="white", mew=0.8, zorder=3)
        ax.annotate(f"{e['difference']:+.3f} [{lo:+.3f}, {hi:+.3f}]", (1.02, y), xycoords=("axes fraction", "data"),
                    va="center", fontsize=7.5, color=MUTED, annotation_clip=False)
        ticks.append(y)
        labels.append(label)
        bold.append(False)
        y += 1
    ax.set_yticks(ticks, labels)
    for tick, b in zip(ax.get_yticklabels(), bold):
        if b:
            tick.set_fontweight("bold")
            tick.set_color(INK)
    ax.invert_yaxis()
    ax.set_xlim(-0.14, 0.10)
    ax.set_xlabel("Difference in AP@0.5 (point estimate and 90% CI)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.text(0, -0.9, "equivalence margin ±0.05", ha="center", fontsize=7.5, color=MUTED)
    fig.savefig(OUT / "results_differences.pdf")
    plt.close(fig)


def figure_froc():
    """FROC sensitivity at the pre-specified false-positive rates, validation and test."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    for ax, (title, get) in zip(axes, (("Validation (BHX)", validation), ("Test (CT-ICH)", test))):
        for arm in ARMS:
            froc = get(arm)["froc_sensitivity"]
            rates = sorted(froc, key=float)
            ax.plot([float(r) for r in rates], [froc[r] for r in rates], color=COLOR[arm], lw=1.6,
                    marker=MARKER[arm], ms=5, mec="white", mew=0.8, label=NAME[arm])
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
        ax.set_xlabel("False positives per exam")
        ax.set_title(title, loc="left", color=INK)
    axes[0].set_ylabel("Box-level sensitivity (IoU ≥ 0.5)")
    axes[0].set_ylim(0, 0.5)
    axes[1].legend(frameon=False, loc="lower right")
    fig.savefig(OUT / "results_froc.pdf")
    plt.close(fig)


def figure_subtypes():
    """Box-level sensitivity per haemorrhage subtype at one false positive per exam (validation)."""
    sens = CORRECTED["validation"]["subtype_sensitivity_1fp"]
    subtypes = sorted(sens["retinanet"], key=lambda s: sens["retinanet"][s]["sensitivity"])
    boxes = {s: sens["retinanet"][s]["boxes"] for s in subtypes}
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    height = 0.25
    for i, arm in enumerate(ARMS):
        y = np.arange(len(subtypes)) + (i - 1) * (height + 0.02)
        ax.barh(y, [sens[arm][s]["sensitivity"] for s in subtypes], height=height, color=COLOR[arm],
                label=NAME[arm], edgecolor="white", linewidth=0.8)
    ax.set_yticks(np.arange(len(subtypes)), [f"{s}\n({boxes[s]:,} boxes)" for s in subtypes])
    ax.set_xlabel("Box-level sensitivity at 1 false positive per exam (validation)")
    ax.set_xlim(0, 0.6)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, loc="lower right")
    fig.savefig(OUT / "results_subtypes.pdf")
    plt.close(fig)


def figure_flow():
    """Flow of examinations from the source datasets to the folds and the test set (counts from the data files)."""
    import csv
    from collections import Counter
    from matplotlib.patches import FancyBboxPatch

    rows = list(csv.DictReader(open(DATA / "cq500_series_escolhidas.csv")))
    group = Counter(r["grupo"] for r in rows)
    with_boxes = sum(r["serie_da_bhx"] == "True" for r in rows)
    without_boxes = len(rows) - with_boxes
    thin, contrast = group["com sangramento, só cortes finos"], group["com sangramento, série com contraste"]
    haem, normal, equivocal, no_thick = (group["com sangramento"], group["normal"], group["duvidoso"],
                                         group["sem série grossa"])
    folds = json.loads((DATA / "export_512_summary.json").read_text())["cq500_folds_main_analysis"]
    fold_sizes = [folds[f"fold{k}"]["exams"] for k in range(1, 6)]
    ctich = json.loads((DATA / "ctich_contagens.json").read_text())
    assert with_boxes == haem + thin + contrast and without_boxes == normal + equivocal + no_thick
    assert sum(fold_sizes) == haem + normal

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 72)
    ax.axis("off")

    def box(x, y, w, h, text, face="white", edge=MUTED, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=face, ec=edge, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6, color=INK, fontweight=weight)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, shrinkA=0, shrinkB=0))

    box(20, 63, 36, 7, f"CQ500: {len(rows)} examinations\n(three readers each)", weight="bold")
    box(2, 49, 30, 7, f"With BHX boxes: {with_boxes}")
    box(44, 49, 30, 7, f"Without BHX boxes: {without_boxes}")
    arrow(30, 63, 17, 56.6)
    arrow(46, 63, 59, 56.6)
    box(2, 37, 30, 8, f"Excluded: {thin + contrast}\nboxes only on thin ({thin})\nor post-contrast ({contrast}) series",
        face="#f4f3f0")
    box(44, 37, 30, 8, f"Excluded: {no_thick}\nno thick non-contrast series", face="#f4f3f0")
    arrow(17, 49, 17, 45.6)
    arrow(59, 49, 59, 45.6)
    box(2, 25, 30, 7, f"With haemorrhage: {haem}")
    box(36, 25, 22, 7, f"Normal: {normal}\n(no reader positive)")
    box(62, 25, 24, 7, f"Equivocal: {equivocal}\n(≥1 reader positive)", face="#f4f3f0")
    arrow(17, 37, 17, 32.6)
    arrow(59, 37, 47, 32.6)
    arrow(59, 37, 74, 32.6)
    ax.text(74, 22.6, "sensitivity analysis only", ha="center", va="top", fontsize=7, color=MUTED, style="italic")
    box(4, 9, 50, 8, f"Main analysis: {haem + normal} examinations\n5 patient-level folds ({min(fold_sizes)}–"
        f"{max(fold_sizes)} examinations each)", face="#eef3fb", edge=COLOR["retinanet"])
    arrow(17, 25, 25, 17.6)
    arrow(47, 25, 37, 17.6)
    box(60, 9, 37, 8, f"CT-ICH test set: {ctich['exames']} examinations\n{ctich['exames_com_caixa']} with, "
        f"{ctich['exames'] - ctich['exames_com_caixa']} without haemorrhage", face="#eef3fb", edge=COLOR["retinanet"])
    ax.text(29, 6.2, "train on 4 folds, validate on the held-out fold\n3 arms × 5 folds = 15 models", ha="center",
            va="top", fontsize=7.2, color=INK)
    ax.text(78.5, 6.2, "predicted by all 15 models\nevaluated once", ha="center", va="top", fontsize=7.2, color=INK)
    fig.savefig(OUT / "study_flow.pdf")
    plt.close(fig)


def latency(arm: str) -> dict:
    """CPU latency of an arm: corrected RetinaNet from notebook 09, YOLO arms from the first run."""
    return (CORRECTED if arm == "retinanet" else FIRST)["cpu_latency"][arm]


def table_summary():
    """Table of the primary and secondary endpoints per arm (LaTeX, written to OUT_DIR/tables)."""
    def ci(p):
        return f"{p['ap50']:.3f} ({p['ci95'][0]:.3f}--{p['ci95'][1]:.3f})"

    drops = CORRECTED["test"]["generalisation_drop"]
    rows = [
        ("\\multicolumn{4}{l}{\\textit{Validation (CQ500/BHX, 5 folds, 436 examinations)}}", None),
        ("AP@0.5 (95\\% CI)", lambda a: ci(validation(a))),
        ("mAP@0.5:0.95", lambda a: f"{validation(a)['map50_95']:.3f}"),
        ("FROC sensitivity at 1 FP/exam", lambda a: f"{validation(a)['froc_sensitivity']['1.0']:.3f}"),
        ("FROC sensitivity at 2 FP/exam", lambda a: f"{validation(a)['froc_sensitivity']['2.0']:.3f}"),
        ("FROC sensitivity at 4 FP/exam", lambda a: f"{validation(a)['froc_sensitivity']['4.0']:.3f}"),
        ("Examination-level AUC", lambda a: f"{validation(a)['exam_auc']:.3f}"),
        ("\\multicolumn{4}{l}{\\textit{Test (CT-ICH, 75 examinations)}}", None),
        ("AP@0.5 (95\\% CI)", lambda a: ci(test(a))),
        ("mAP@0.5:0.95", lambda a: f"{test(a)['map50_95']:.3f}"),
        ("FROC sensitivity at 1 FP/exam", lambda a: f"{test(a)['froc_sensitivity']['1.0']:.3f}"),
        ("Examination-level AUC", lambda a: f"{test(a)['exam_auc']:.3f}"),
        ("Drop in AP@0.5 from validation (95\\% CI)",
         lambda a: f"{drops[a]['drop']:.3f} ({drops[a]['ci95'][0]:.3f}--{drops[a]['ci95'][1]:.3f})"),
        ("\\multicolumn{4}{l}{\\textit{Inference on CPU (fold-1 model, 100 CT-ICH slices)}}", None),
        ("Median time per slice (ms)", lambda a: f"{latency(a)['median_seconds_per_slice'] * 1000:.0f}"),
        ("90th percentile (ms)", lambda a: f"{latency(a)['p90_seconds_per_slice'] * 1000:.0f}"),
    ]
    lines = ["% Generated by tools/make_figures.py from the result files; do not edit by hand.",
             "\\begin{table*}[pos=t]", "\\centering",
             "\\caption{Primary and secondary endpoints per arm. Cross-validation values are means over the five "
             "fold models evaluated on their own held-out examinations; test values are means of the same five "
             "models on CT-ICH. Confidence intervals are percentile bootstrap intervals over examinations "
             "(10,000 resamples). FP, false positives; AUC, area under the ROC curve. CPU times of the YOLOv8m "
             "arms were measured in the first run.}",
             "\\label{tab:results}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{p{0.24\\textwidth}ccc}", "\\toprule",
             " & RetinaNet & YOLOv8m & YOLOv8m \\\\", " & (arm A) & no mosaic (B) & mosaic (C) \\\\", "\\midrule"]
    for label, get in rows:
        lines.append(f"{label} \\\\" if get is None else f"{label} & " + " & ".join(get(a) for a in ARMS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    out = OUT.parent / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results_summary.tex").write_text("\n".join(lines) + "\n")


def table_first_run():
    """Supplementary table: every endpoint of the RetinaNet trained with transposed boxes and of the corrected one."""
    runs = {"first": (FIRST["validation"]["primary_A_vs_B"]["A"], FIRST["test"]["A_vs_B"]["A"],
                      FIRST["test"]["generalisation_drop"]["retinanet"], FIRST["cpu_latency"]["retinanet"]),
            "corrected": (validation("retinanet"), test("retinanet"),
                          CORRECTED["test"]["generalisation_drop"]["retinanet"], latency("retinanet"))}

    def ci(p):
        return f"{p['ap50']:.3f} ({p['ci95'][0]:.3f}--{p['ci95'][1]:.3f})"

    rows = []
    for setting, k in (("Cross-validation", 0), ("Test (CT-ICH)", 1)):
        rows.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{{setting}}}}} \\\\")
        rows.append("AP@0.5 (95\\% CI) & " + " & ".join(ci(r[k]) for r in runs.values()) + " \\\\")
        rows.append("mAP@0.5:0.95 & " + " & ".join(f"{r[k]['map50_95']:.3f}" for r in runs.values()) + " \\\\")
        for fp in ("0.5", "1.0", "2.0", "4.0"):
            rows.append(f"FROC sensitivity at {float(fp):g} FP/exam & "
                        + " & ".join(f"{r[k]['froc_sensitivity'][fp]:.3f}" for r in runs.values()) + " \\\\")
        rows.append("Examination-level AUC & " + " & ".join(f"{r[k]['exam_auc']:.3f}" for r in runs.values()) + " \\\\")
    rows.append("Drop in AP@0.5 from validation (95\\% CI) & " + " & ".join(
        f"{r[2]['drop']:.3f} ({r[2]['ci95'][0]:.3f}--{r[2]['ci95'][1]:.3f})" for r in runs.values()) + " \\\\")
    rows.append("CPU time per slice, median (ms) & " + " & ".join(
        f"{r[3]['median_seconds_per_slice'] * 1000:.0f}" for r in runs.values()) + " \\\\")
    lines = ["% Generated by tools/make_figures.py from the result files; do not edit by hand.",
             "\\begin{table}[ht]", "\\centering",
             "\\caption{RetinaNet trained with transposed boxes (first run, protocol v1.1) and corrected RetinaNet "
             "(protocol v1.2): all endpoints. Values as in the main text: means over the five fold models, 95\\% "
             "percentile bootstrap intervals over examinations.}",
             "\\label{tab:first-run}", "\\small", "\\begin{tabular}{p{0.42\\textwidth}cc}", "\\toprule",
             " & First run & Corrected \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT.parent / "tables" / "first_run_retinanet.tex").write_text("\n".join(lines) + "\n")


def figure_curves():
    """Exploratory: exam-level ROC and box-level precision-recall (IoU 0.5), validation and test.

    Thin lines are the five fold models; thick lines their vertical mean (the mean of the precision curve
    is the mean AP@0.5 reported in the text).
    """
    fpr = np.asarray(EXPLORATORY["roc_fpr_grid"])
    rec = np.asarray(EXPLORATORY["recall_points"])
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.6), sharex="row", sharey="row")
    for col, (setting, title) in enumerate((("validation", "Validation (BHX)"), ("test", "Test (CT-ICH)"))):
        curves = EXPLORATORY["curves"][setting]
        for arm in ARMS:
            for row, (key, x) in enumerate((("roc_exam", fpr), ("pr_iou50", rec))):
                ax = axes[row, col]
                ys = np.asarray(curves[arm][key])
                for y in ys:
                    ax.plot(x, y, color=COLOR[arm], lw=0.5, alpha=0.3, drawstyle="steps-post" if row == 0 else "default")
                metric = "exam_auc" if row == 0 else "ap50"
                value = EXPLORATORY[setting]["arms"][arm][metric]["value"]
                ax.plot(x, ys.mean(axis=0), color=COLOR[arm], lw=1.6,
                        label=f"{NAME[arm]} ({'AUC' if row == 0 else 'AP'} {value:.3f})")
        axes[0, col].plot([0, 1], [0, 1], color=GREY, lw=0.8, ls=(0, (3, 3)))
        axes[0, col].set_title(title, loc="left", color=INK)
        axes[0, col].set_xlabel("1 − specificity (examinations)")
        axes[1, col].set_xlabel("Recall (boxes, IoU ≥ 0.5)")
        for row in (0, 1):
            axes[row, col].legend(frameon=False, loc="lower right" if row == 0 else "upper right", fontsize=7)
    axes[0, 0].set_ylabel("Sensitivity (examinations)")
    axes[1, 0].set_ylabel("Precision (boxes)")
    for ax in axes.flat:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(OUT / "exploratory_curves.pdf")
    plt.close(fig)


def figure_loss():
    """Exploratory: training loss per epoch of every fold model (losses differ between detectors)."""
    labels = {"retinanet": "focal + smooth L1", "yolo_no_mosaic": "box + cls + DFL", "yolo_mosaic": "box + cls + DFL"}
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4))
    for ax, arm in zip(axes, ARMS):
        for fold, h in EXPLORATORY["training_loss"][arm].items():
            ax.plot(h["epoch"], h["loss"], color=COLOR[arm], lw=0.9, alpha=0.75)
        if arm == "yolo_mosaic":
            ax.axvspan(40.5, 50, color=GREY, alpha=0.15, lw=0)
        ax.set_title(NAME[arm], loc="left", color=INK, fontsize=8.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Training loss ({labels[arm]})", fontsize=7.5)
        ax.set_xlim(1, 50)
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT / "exploratory_loss.pdf")
    plt.close(fig)


def table_exploratory():
    """Supplementary table: exploratory endpoints with 95% CIs, and the paired difference RetinaNet − YOLOv8m."""
    names = [("ap50", "AP@0.5 (primary, for reference)"), ("ap75", "AP@0.75"), ("map50_95", "mAP@0.5:0.95"),
             ("exam_auc", "Examination-level AUC"), ("precision_1fp", "Box precision at 1 FP/exam"),
             ("recall_1fp", "Box recall at 1 FP/exam"), ("f1_1fp", "Box F1 at 1 FP/exam"),
             ("exam_sensitivity_1fp", "Examination sensitivity at 1 FP/exam"),
             ("exam_specificity_1fp", "Examination specificity at 1 FP/exam")]

    def cell(m, signed=False):
        if signed:
            return f"{m['value']:+.3f} ({m['ci95'][0]:+.3f} to {m['ci95'][1]:+.3f})"
        return f"{m['value']:.3f} ({m['ci95'][0]:.3f}--{m['ci95'][1]:.3f})"

    rows = []
    for setting, title in (("validation", "Cross-validation (CQ500/BHX, 436 examinations)"),
                           ("test", "Test (CT-ICH, 75 examinations)")):
        r = EXPLORATORY[setting]
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for key, label in names:
            rows.append(f"{label} & " + " & ".join(cell(r["arms"][a][key]) for a in ARMS) + " & "
                        + cell(r["differences"]["retinanet_minus_yolo_no_mosaic"][key], True) + " \\\\")
        sizes = EXPLORATORY["size_sensitivity"][setting]
        for size, rule in (("small", "$<32^2$"), ("medium", "$32^2$--$96^2$"), ("large", "$\\geq 96^2$")):
            rows.append(f"Recall at 1 FP/exam, {size} boxes ({rule} px) & " + " & ".join(
                f"{sizes[a][size]['sensitivity']:.3f} ({sizes[a][size]['detected']}/{sizes[a][size]['boxes']})"
                for a in ARMS) + " & -- \\\\")
    lines = ["% Generated by tools/make_figures.py from the result files; do not edit by hand.",
             "\\begin{table}[ht]", "\\centering",
             "\\caption{Exploratory endpoints, added after the registered analysis. Means over the five fold models "
             "with 95\\% percentile bootstrap intervals over examinations (10,000 paired resamples); the last "
             "column is the paired difference RetinaNet $-$ YOLOv8m without mosaic. Each fold model uses its own "
             "operating point of one false positive (FP) per examination. Box sizes follow the COCO area ranges "
             "on the $512 \\times 512$ input (32~px $\\approx$ 15.6~mm); size counts are summed over the five fold models, so that on CT-ICH each of the 397 reference boxes is counted once per model. The bootstrap draws differ from those of Table~4 of the main text, so AP@0.5 intervals can differ in the third decimal place.}",
             "\\label{tab:exploratory}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}p{0.25\\textwidth}cccc@{}}", "\\toprule",
             " & RetinaNet & YOLOv8m & YOLOv8m & Difference \\\\",
             " & (A) & no mosaic (B) & mosaic (C) & A $-$ B \\\\", "\\midrule", *rows,
             "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT.parent / "tables" / "exploratory_endpoints.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure_performance()
    figure_differences()
    figure_froc()
    figure_subtypes()
    figure_flow()
    table_summary()
    table_first_run()
    figure_curves()
    figure_loss()
    table_exploratory()
    print("figures written to", OUT)
