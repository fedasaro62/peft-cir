"""Cross-dataset transfer results: who wins where, as bars.

The one figure script this study keeps. It answers the question the cross-dataset matrix
exists for -- when a checkpoint adapted on one benchmark is scored unmodified on another,
does adaptation still beat doing nothing, and which method wins -- and it draws that as a
grid of (source -> target) x architecture cells.

Three bars per cell. Frozen always occupies row 1, fixed there regardless of how it ranks, so
it is always the same place to look; then the top TOP_N PEFT methods, best first. Bold marks
whichever of the three is highest, **Frozen included** -- a cell where adaptation fails to beat
the zero-shot baseline shows its bold on the Frozen row, which is exactly the case worth
spotting at a glance, and the box tints grey rather than by the leading PEFT method. The method
name is written on its own row, so nothing needs decoding against a legend.

All three bars run from one shared local floor to that row's own value, so every bar answers
"how big" on the same scale and a row's value never needs a second reading against another row.
The floor is always padded below the group's true minimum, so the weakest bar shown is never a
zero-length sliver merely for being the smallest of three. Coverage is partial by design while
jobs are still running: the caption states the cell count out of the full matrix, and a cell
with fewer than MIN_METHODS_FOR_PANEL landed methods is held back rather than ranked thin.

Two outputs, from one run:

  peft_cross_results          all nine (source, target) pairs together -- "how complete is the
                              study, and who wins overall"
  peft_cross_results_{a..d}   the same grid partitioned by how the two domains relate, which is
                              the structural question the combined view buries:
                                (A) in-domain, general   CIRR->CIRCO
                                (B) in-domain, fashion   FIQ<->Shoes
                                (C) narrowing            CIRR->{FIQ, Shoes}
                                (D) broadening           {FIQ, Shoes}->{CIRR, CIRCO}
                              The four groups cover ALL_PAIRS exactly once (1+2+2+4 = 9), so
                              this is a partition of the combined grid, not a new selection.
                              (C) is CIRR-only because CIRCO is eval-only throughout the study,
                              so CIRCO->FIQ and CIRCO->Shoes cannot exist by design, not merely
                              "not yet landed".

The four panels are transposed (architectures on the columns, pairs on the rows) so the axis
with more items is always the columns: each is then the same width and reads wide rather than
as a narrow strip once LaTeX stretches it to \\linewidth. They are drawn on white rather than
the study's off-white SURFACE, because they sit directly on the white page where #fcfcfb reads
as a visible grey rectangle.

    PYTHONPATH=src .venv/lincir/bin/python tools/plot_cross_results.py
"""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIGDIR, DOCDIR = ROOT / "outputs" / "figures", ROOT / "docs"

# Colour carries the adapter cost tier (categorical slots 1-2 of the validated palette, used
# unmodified -- all-pairs CVD dE 24.7, normal-vision 33.6, contrast 4.30/3.12, all clear in
# light mode). Frozen wears the de-emphasis grey because it is a baseline, not a third tier.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, GRID, INK, INK2, MUTED = "#fcfcfb", "#e3e2dc", "#0b0b0b", "#52514e", "#898781"
TIER = {"lora": BLUE, "dora": BLUE, "adapter": BLUE, "adalora": BLUE,
        "ia3": ORANGE, "vera": ORANGE, "full": MUTED, "frozen": MUTED}

# the study's four architectures, in the order they are drawn: the two textual-inversion
# retrievers first, then the two that retrieve through a fused/summed representation
MODEL_LABEL = {"pic2word": "Pic2Word", "searle": "SEARLE-XL",
               "magiclens": "MagicLens", "mti": "MTI"}
PEFT_METHODS = ["lora", "dora", "ia3", "adapter", "adalora", "vera"]  # ranked; Full excluded
MET_LABEL = {"lora": "LoRA", "dora": "DoRA", "ia3": "(IA)$^3$", "adapter": "Adapter",
             "adalora": "AdaLoRA", "vera": "VeRA", "full": "Full FT"}
# FIQ abbreviates FashionIQ, matching the write-up's own column-header convention
TRAIN_LABEL = {"cirr": "CIRR", "fiq": "FashionIQ", "shoes": "Shoes"}
TARGET_LABEL = {**TRAIN_LABEL, "circo": "CIRCO"}
SHORT_TRAIN = {**TRAIN_LABEL, "fiq": "FIQ"}
SHORT_TARGET = {**TARGET_LABEL, "fiq": "FIQ"}

# every (train, target) pair the study runs: 3 train sets x 3 off-diagonal targets
ALL_PAIRS = [(tr, tg) for tr in ("cirr", "fiq", "shoes")
             for tg in ("cirr", "fiq", "shoes", "circo") if tg != tr]
CROSS_TOTAL_CELLS = len(ALL_PAIRS) * len(MODEL_LABEL)
# a cell with only one or two methods landed is a near-empty ranking, not an informative one;
# hold it back until it clears this bar rather than rank whatever happens to exist
MIN_METHODS_FOR_PANEL = 4

# (group key, label, pairs) -- label is used for the console summary only
GROUPS = [
    ("a", "in-domain, general", [("cirr", "circo")]),
    ("b", "in-domain, fashion", [("fiq", "shoes"), ("shoes", "fiq")]),
    ("c", "domain narrowing",   [("cirr", "fiq"), ("cirr", "shoes")]),
    ("d", "domain broadening",  [("fiq", "cirr"), ("fiq", "circo"),
                                 ("shoes", "cirr"), ("shoes", "circo")]),
]

# (target benchmark, split) for the in-domain frozen reference. CIRR/FashionIQ/CIRCO report
# val (their test ground truth is server-side); Shoes reports test under the literature
# protocol, matching every other Shoes number in this study. Frozen rows always live in the
# plain {model}_peft tree: a frozen run involves no training, so it is never attributed to
# a recipe's own tree, for any architecture.
FROZEN_SRC = {
    "cirr":  ("CIRR/{model}_peft", "cirr", "val"),
    "fiq":   ("fashioniq/{model}_peft", "fashioniq-avg", "val"),
    "shoes": ("shoes/{model}_peft", "shoes", "test"),
    "circo": ("CIRCO/{model}_peft", "circo", "val"),
}
# cross-dataset benchmarks.csv location, per model: the textual-inversion pair keeps every
# method in one "{model}_peft" dir; magiclens and mti write their cross-eval rows under the
# tree named for the recipe they train with, not the plain one
CROSS_DIR = {"magiclens": "magiclens_peft_faithful", "mti": "mti_peft_asymmetric"}
# suffix a text+vision row's method column carries: magiclens/mti tag every run with the
# asymmetric loss they train under; pic2word/searle carry none
TEXT_VISION_SUFFIX = {"magiclens": "_text_vision_asymmetric", "mti": "_text_vision_asymmetric"}

TOP_N = 2  # PEFT-method bars per cell, below the fixed Frozen row
DPI = 200
LABEL_FS, TITLE_FS, ROWNAME_FS, EMPTY_FS, LEGEND_FS, METHOD_FS = 11, 12, 13, 9.5, 11, 10
ROW_YS = (0.68, 0.50, 0.32)  # Frozen, rank 1, rank TOP_N -- local y-fractions (of a cell)
BOX_PAD = 0.035  # inset between a box and the next cell's box (whitespace gutter), each side
BOX_H = 0.82     # box height, local y-fraction -- hugs the three bars vertically
INSET = 0.03     # inset from the box's own edge to the first drawn element
BAR_H = 0.11     # bar thickness, local y-fraction (~24px at this DPI/cell_h -- the spec's cap)
METHOD_GAP_IN = 0.09   # gap between the method-name column and the bar's travel zone
VALUE_GAP_IN = 0.05    # gap between a bar's tip and its value label
BAR_ZONE_IN = 0.38     # the bar's own travel range at width_scale=1 -- a layout choice, not a
                       # measured quantity; render() multiplies it by width_scale


def avg_score(target: str, row: pd.Series) -> float:
    """Each benchmark's own selection-metric Avg.

    CIRCO has no selection metric anywhere in this codebase -- it was only ever an eval-only
    cross-dataset target, never trained or selected on -- so its rows carry mAP@k, not
    Recall@k, and would silently average to NaN under the Recall formula below. Defined here
    as (mAP@5 + mAP@10)/2, mirroring every other benchmark's own convention of averaging its
    two most commonly reported cutoffs.
    """
    if target == "cirr":
        return (row["Recall@5"] + row["Recall_subset@1"]) / 2
    if target == "circo":
        return (row["mAP@5"] + row["mAP@10"]) / 2
    return (row["Recall@10"] + row["Recall@50"]) / 2  # fiq's row is already subtask-pooled


def train_of(model_tag: str) -> str | None:
    """Which benchmark a checkpoint's model tag says it was trained on."""
    # magiclens's "magiclens_large_cirr_faithful" / "..._shoes_faithful_lit" tags carry an
    # extra "_faithful" marker the others don't; strip both optional suffixes so one pattern
    # covers every architecture's naming
    tag = model_tag.removesuffix("_lit").removesuffix("_faithful")
    return next((t for t in ("cirr", "fiq", "shoes") if tag.endswith(f"_{t}")), None)


def load_cross(model: str) -> dict:
    """(train, target) -> {method: avg}, from every landed text+vision cross-eval row."""
    csv = FIGDIR.parent / "cross" / CROSS_DIR.get(model, f"{model}_peft") / "benchmarks.csv"
    if not csv.exists():
        return {}
    suffix = TEXT_VISION_SUFFIX.get(model, "_text_vision")
    df = pd.read_csv(csv)
    df = df[df["method"].str.endswith(suffix)]
    out: dict = {}
    for _, r in df.iterrows():
        train = train_of(r["model"])
        target = r["benchmark"].split("-")[0]
        target = "fiq" if target == "fashioniq" else target
        if train is None or train == target:
            continue
        out.setdefault((train, target), {})[r["method"].removesuffix(suffix)] = avg_score(target, r)
    return out


def load_frozen(model: str) -> dict:
    """Target benchmark -> zero-shot Avg for this model; the no-training reference bar."""
    out = {}
    for target, (subdir, bench, split) in FROZEN_SRC.items():
        candidates = [subdir.format(model=model)]
        if target == "shoes":
            # mti has no plain outputs/shoes/mti_peft dir at all -- its frozen Shoes row lives
            # only under the literature-protocol dir, unlike every other (model, target) pair
            candidates.append(f"shoes/{model}_peft_lit")
        for cand in candidates:
            csv = FIGDIR.parent / cand / "benchmarks.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            # mti's benchmarks.csv also carries a "mti_stock_large" plain-CLIP control row
            # under the same method/split/benchmark -- exclude it, so the frozen reference is
            # always the masked-tuned checkpoint and not whichever row happens to load first
            row = df[(df["method"] == "frozen") & (df["split"] == split)
                     & (df["benchmark"] == bench)
                     & (~df["model"].str.contains("stock", na=False))]
            if len(row):
                out[target] = avg_score(target, row.iloc[0])
                break
    return out


def hold_back_thin_cells(by_model: dict) -> dict:
    """Drop (model, cell)s with fewer than MIN_METHODS_FOR_PANEL landed PEFT methods."""
    filtered, thin_report = {}, []
    for model, cells in by_model.items():
        kept, thin = {}, []
        for pair, methods in cells.items():
            peft_only = {m: v for m, v in methods.items() if m in PEFT_METHODS}
            if len(peft_only) >= MIN_METHODS_FOR_PANEL:
                kept[pair] = peft_only
            else:
                thin.append(pair)
        if thin:
            thin_report.append(f"{model}: {sorted(thin, key=ALL_PAIRS.index)}")
        if kept:
            filtered[model] = kept
    if thin_report:
        print("holding back (too few methods landed yet):", "; ".join(thin_report))
    return filtered


def load_all() -> tuple[dict, dict]:
    """(filtered, frozen), loaded once from outputs/cross/ -- shared by every render() call."""
    by_model = {m: load_cross(m) for m in MODEL_LABEL}
    by_model = {m: cells for m, cells in by_model.items() if cells}
    frozen = {m: load_frozen(m) for m in by_model}
    return hold_back_thin_cells(by_model), frozen


def measure_text_width_in(strings, fontsize: float) -> float:
    """max rendered width (inches), at this exact fontsize/DPI, of any of ``strings``.

    Renders each candidate for real and reads back its pixel bounding box -- the only way
    to know a mathtext string's ("(IA)$^3$") true width -- rather than guess from character
    count, which is what silently overflowed the box before this was measured for real.
    """
    probe_fig = plt.figure(dpi=DPI)
    probe_ax = probe_fig.add_axes([0, 0, 1, 1])
    probe_fig.canvas.draw()
    renderer = probe_fig.canvas.get_renderer()
    widths = []
    for s in strings:
        t = probe_ax.text(0, 0, s, fontsize=fontsize)
        probe_fig.canvas.draw()
        widths.append(t.get_window_extent(renderer=renderer).width / DPI)
    plt.close(probe_fig)
    return max(widths)


def measure_legend_height_in(handles, ncol: int, fontsize: float) -> float:
    """rendered height (inches) of a figure-level legend with these handles, measured the
    same way as measure_text_width_in -- so the space reserved for it below the grid is
    never a guess either."""
    probe_fig = plt.figure(dpi=DPI)
    legend = probe_fig.legend(handles=handles, loc="center", ncol=ncol, frameon=False,
                              fontsize=fontsize)
    probe_fig.canvas.draw()
    h_in = legend.get_window_extent(renderer=probe_fig.canvas.get_renderer()).height / DPI
    plt.close(probe_fig)
    return h_in


def render(filtered: dict, frozen: dict, model_order: list, pairs: list, out_name: str,
          width_scale: float = 1.0, total_cells: int = None, transpose: bool = False,
          background: str = SURFACE) -> None:
    """draw the grid for exactly ``pairs`` x ``model_order``, writing
    outputs/figures/{out_name}.png, and a copy to docs/figs/ when that directory exists.

    Columns are ``pairs`` and rows are ``model_order`` by default, matching the combined
    figure. ``transpose=True`` swaps that -- columns become ``model_order`` (always 4 here)
    and rows become ``pairs`` -- which is what the per-category panels want for their 1- and
    2-pair groups: the axis with *more* items should be the columns, so the figure reads wide
    rather than as a tall, narrow strip once LaTeX stretches it to \\linewidth. Every per-cell
    size (bars, fonts, box padding) is identical either way -- only which label sits on top vs.
    on the left changes.

    ``background`` paints the figure, the axes and the saved file. It defaults to the study's
    off-white SURFACE so the combined figure matches every other one; the per-category panels
    pass pure white, since they are placed directly on white pages.

    ``width_scale`` multiplies BAR_ZONE_IN -- and only that -- for further, finer width control
    on top of ``transpose`` if a selection still needs it. ``total_cells`` is the "N of ..."
    denominator printed at the end; default is this selection's own size
    (len(pairs) * len(model_order)), not the whole study's.
    """
    if not pairs or not model_order:
        print(f"{out_name}: nothing to plot")
        return
    bar_zone_in = BAR_ZONE_IN * width_scale
    n_landed = sum(1 for m in model_order for p in pairs if p in filtered.get(m, {}))
    total_cells = total_cells if total_cells is not None else len(pairs) * len(model_order)

    def pair_title(pair: tuple) -> str:
        tr, tg = pair
        return f"{SHORT_TRAIN[tr]}$\\to${SHORT_TARGET[tg]}"

    if transpose:
        col_items, row_items = model_order, pairs
        col_label, row_label = (lambda m: MODEL_LABEL[m]), pair_title
    else:
        col_items, row_items = pairs, model_order
        col_label, row_label = pair_title, (lambda m: MODEL_LABEL[m])

    # every label this run will actually draw, so every measured max is never a stale guess.
    # "--" covers the handful of cells with no landed Frozen eval (see module docstring)
    cells_in_view = [filtered[m][p] for m in model_order for p in pairs if p in filtered.get(m, {})]
    all_labels = ["--"] + [f"{v:.1f}" for cell in cells_in_view for v in list(cell.values())[:TOP_N]] \
        + [f"{v:.1f}" for m in model_order for v in frozen.get(m, {}).values()]
    value_zone_in = measure_text_width_in(all_labels, LABEL_FS) * 1.08  # 8% safety margin
    method_zone_in = measure_text_width_in(["Frozen"] + [MET_LABEL[m] for m in PEFT_METHODS],
                                           METHOD_FS) * 1.08
    row_label_in = measure_text_width_in([row_label(r) for r in row_items], ROWNAME_FS) * 1.08

    # method-name column, a gap, the bar's travel zone, a gap, then the value label -- one
    # value-label margin, not two, since every bar now grows the same direction (see docstring)
    content_in = method_zone_in + METHOD_GAP_IN + bar_zone_in + VALUE_GAP_IN + value_zone_in
    box_cell_w = content_in / (1 - 2 * BOX_PAD - 2 * INSET)
    # a column's title sits centered on one cell and, once the boxes shrank to fit numbers
    # alone, became the wider of the two constraints -- found by the titles overlapping
    # into an unreadable run when this was left out and cell_w came from the boxes only
    title_cell_w = measure_text_width_in([col_label(c) for c in col_items], TITLE_FS) * 1.15
    cell_w = max(box_cell_w, title_cell_w)

    METHOD_X = BOX_PAD + INSET
    BAR_X0 = METHOD_X + (method_zone_in + METHOD_GAP_IN) / cell_w
    BAR_X1 = BAR_X0 + bar_zone_in / cell_w

    def local_x(value: float, lo: float, hi: float) -> float:
        """value -> a local x-fraction in [BAR_X0, BAR_X1], scaled by this cell's own range."""
        frac = 0.5 if hi == lo else (value - lo) / (hi - lo)
        return BAR_X0 + frac * (BAR_X1 - BAR_X0)

    # the legend is figure-level, not axes-level, so its height has to be reserved as its own
    # band below the grid; anchoring it via a negative bbox_to_anchor fraction leaves that gap
    # a guess, and measuring the actual legend and reserving 1.3x its height makes it correct
    # by construction instead. Method identity is the row's own text, not a legend lookup, so
    # the seven-shape key v1 needed is gone -- just the two cost-tier fills plus Frozen's own.
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=13, color=MUTED,
                          label="Frozen"),
              plt.Line2D([], [], marker="s", linestyle="", markersize=13, color=TIER["lora"],
                        label="expressive adapter"),
              plt.Line2D([], [], marker="s", linestyle="", markersize=13, color=TIER["ia3"],
                        label="minimal adapter")]
    legend_band_in = measure_legend_height_in(handles, len(handles), LEGEND_FS) * 1.3

    n_rows, n_cols = len(row_items), len(col_items)
    cell_h = 0.92
    left_margin_in = row_label_in + 0.35
    grid_h_in = cell_h * n_rows + 0.7
    fig_h_in = grid_h_in + legend_band_in
    fig, ax = plt.subplots(figsize=(cell_w * n_cols + left_margin_in, fig_h_in))
    # every width above (cell_w, title_cell_w, ...) is an inches-per-data-unit assumption;
    # matplotlib's default subplot margins would leave the axes narrower than the figure and
    # silently violate that assumption -- filling the raw canvas exactly keeps the two in
    # lockstep. bbox_inches="tight" still crops the saved file to the real content afterwards.
    fig.subplots_adjust(left=0, right=1, top=1, bottom=legend_band_in / fig_h_in)
    fig.set_facecolor(background)
    ax.set_facecolor(background)

    def draw_bar(col: float, ry: float, label: str, value, color: str, lo: float, hi: float,
                bold: bool = False) -> None:
        """one row: the method/Frozen name at a fixed left column, a bar from this cell's
        local floor to ``value``, and the value written past the bar's own tip. ``value`` is
        None for a Frozen row with no landed eval (draws the label and a bare "--")."""
        ax.text(col + METHOD_X, ry, label, ha="left", va="center", fontsize=METHOD_FS,
                color=INK if bold else INK2, fontweight="semibold" if bold else "normal", zorder=5)
        if value is None:
            ax.text(col + BAR_X0, ry, "--", ha="left", va="center", fontsize=LABEL_FS,
                    color=INK2, zorder=5)
            return
        x0 = col + local_x(lo, lo, hi)
        x1 = col + local_x(value, lo, hi)
        bar_w = max(x1 - x0, 0.006)  # the floor's own row (bar_w -> 0) still stays visible
        ax.add_patch(FancyBboxPatch((x0, ry - BAR_H / 2), bar_w, BAR_H,
                                    boxstyle="round,pad=0,rounding_size=0.012",
                                    linewidth=0, facecolor=color, zorder=4))
        ax.text(x1 + VALUE_GAP_IN / cell_w, ry, f"{value:.1f}", ha="left", va="center",
                fontsize=LABEL_FS, color=INK, zorder=5)

    for col, item in enumerate(col_items):
        ax.text(col + 0.5, n_rows + 0.15, col_label(item), ha="center", va="bottom",
                fontsize=TITLE_FS, color=INK, fontweight="semibold")

    for row, row_item in enumerate(row_items):
        y = n_rows - row - 1  # row 0 drawn at the top
        ax.text(-0.15, y + 0.5, row_label(row_item), ha="right", va="center",
                fontsize=ROWNAME_FS, color=INK, fontweight="semibold")
        for col, col_item in enumerate(col_items):
            model, pair = (col_item, row_item) if transpose else (row_item, col_item)
            cell = filtered.get(model, {}).get(pair)
            if cell is None:
                ax.add_patch(FancyBboxPatch((col + 0.06, y + 0.5 - BOX_H / 2), 0.88, BOX_H,
                                            boxstyle="round,pad=0,rounding_size=0.04",
                                            linewidth=0.9, linestyle=(0, (3, 2)),
                                            edgecolor=GRID, facecolor="none"))
                ax.text(col + 0.5, y + 0.5, "not yet\nlanded", ha="center", va="center",
                        fontsize=EMPTY_FS, color=MUTED)
                continue

            ranked = sorted(cell.items(), key=lambda kv: -kv[1])[:TOP_N]
            fz = frozen.get(model, {}).get(pair[1])
            raw_vals = [v for _, v in ranked] + ([fz] if fz is not None else [])
            # the local floor is always padded below the true minimum -- not only when Frozen
            # is missing -- so the weakest bar shown is never a ~0-length sliver merely for
            # being the smallest of the two or three values in this cell (see module docstring)
            span = max(raw_vals) - min(raw_vals)
            lo = min(raw_vals) - max(span, 1.0) * 0.4
            hi = max(raw_vals)

            # bold marks this cell's BEST of the three rows drawn, Frozen included -- not
            # simply the leading PEFT method. `ranked` is sorted descending, so the winner is
            # either Frozen or rank 1; a tie leaves the bold on the PEFT row, since Frozen has
            # to strictly beat adaptation to claim it. The box fill follows the same winner --
            # grey (Frozen's own colour) when adaptation fails to beat the zero-shot baseline,
            # rather than always tinting by the leading PEFT method regardless of who actually
            # won the cell.
            frozen_wins = fz is not None and fz > ranked[0][1]
            winner_tier = MUTED if frozen_wins else TIER[ranked[0][0]]
            ax.add_patch(FancyBboxPatch((col + 0.06, y + 0.5 - BOX_H / 2), 0.88, BOX_H,
                                        boxstyle="round,pad=0,rounding_size=0.04",
                                        linewidth=1.1, edgecolor=winner_tier,
                                        facecolor=winner_tier, alpha=0.13))
            draw_bar(col, y + ROW_YS[0], "Frozen", fz, MUTED, lo, hi, bold=frozen_wins)
            for i, ((method, value), ry_frac) in enumerate(zip(ranked, ROW_YS[1:])):
                draw_bar(col, y + ry_frac, MET_LABEL[method], value, TIER[method], lo, hi,
                        bold=(i == 0 and not frozen_wins))

    ax.set_xlim(-left_margin_in / cell_w, n_cols)
    ax.set_ylim(0, n_rows + 0.55)
    ax.axis("off")

    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
              labelcolor=INK, handletextpad=0.35, columnspacing=1.3, fontsize=LEGEND_FS,
              bbox_to_anchor=(0.5, 0))

    # PNG only: it is what the write-up includes, and outputs/ is regenerated, so a parallel
    # PDF was only ever a second copy of the same picture
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{out_name}.png", dpi=DPI, facecolor=background,
               bbox_inches="tight", pad_inches=0.06)
    # a second copy under docs/figs/, for a write-up whose \includegraphics paths resolve
    # there; skipped when there is no docs/ directory to copy into
    if DOCDIR.is_dir():
        docs_figdir = DOCDIR / "figs"
        docs_figdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIGDIR / f"{out_name}.png", docs_figdir / f"{out_name}.png")
    plt.close(fig)
    print(f"{out_name}: cells plotted: {pairs}  (models: {model_order})")
    print(f"{out_name}: {n_landed} of {total_cells} (model, source, target) cells")
    print(f"{out_name}: method zone: {method_zone_in:.2f}in, value zone: {value_zone_in:.2f}in, "
         f"cell width: {cell_w:.2f}in")
    print(f"wrote {FIGDIR}/{out_name}.png")



def main() -> None:
    filtered, frozen = load_all()
    if not filtered:
        print("no cell has enough landed methods yet; nothing to plot")
        return

    pairs = sorted({p for c in filtered.values() for p in c}, key=ALL_PAIRS.index)
    model_order = [m for m in MODEL_LABEL if m in filtered]
    missing = [m for m in MODEL_LABEL if m not in filtered]
    if missing:
        print(f"note: no landed cells for {missing} -- omitted")
    render(filtered, frozen, model_order, pairs, "peft_cross_results",
           total_cells=CROSS_TOTAL_CELLS)

    # the per-category panels keep ALL four columns even where a group has no landed cell for
    # one of them: draw_bar's "not yet landed" placeholder already covers that, and a column
    # silently vanishing between panels would misread as a model being excluded
    for key, label, group_pairs in GROUPS:
        print(f"--- group ({key.upper()}) {label}: {group_pairs} ---")
        render(filtered, frozen, list(MODEL_LABEL), group_pairs, f"peft_cross_results_{key}",
               transpose=True, background="white")


if __name__ == "__main__":
    main()
