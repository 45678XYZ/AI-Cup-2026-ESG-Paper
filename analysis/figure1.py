"""Figure 1: the label hierarchy and the alternative decision routes.

Plan section 5 asks one figure to carry four things: the four-field dependency
and its 17 legal states, the base probabilities, the two calibration options,
and the three output rules.

It also forbids one thing, and that is what drives the layout: this must not
look like a four-layer architecture with every module switched on. The main
experiment compares mutually exclusive routes, so calibration and decoding are
drawn as parallel branches labelled with the method ids that take them, never
as stacked stages.

Boxes are positioned by centre and the drawn outline extends ``PAD`` beyond the
requested rectangle on every side -- ``FancyBboxPatch`` pads outwards, so
column spacing has to budget for it or neighbouring boxes silently merge into
what looks like a single rule.

Vector PDF with fonts embedded as Type 42, per contract section 5.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType, not Type 3
matplotlib.rcParams["font.size"] = 7.0

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from paper.labels import STATES  # noqa: E402

PAD = 0.18
ROUTE_COLOURS = {"none": "#4d4d4d", "global": "#1f77b4", "conditional": "#d62728"}
GREY = "#8c8c8c"
INK = "#333333"


def _box(ax, cx, cy, w, h, text, *, edgecolor=INK):
    """Draw a box centred on ``(cx, cy)``.

    Returns ``(bottom, top)`` anchor points that already include ``PAD``, so
    arrows meet the drawn outline rather than the un-padded rectangle.
    """
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={PAD}", linewidth=0.8,
        facecolor="white", edgecolor=edgecolor,
    ))
    ax.text(cx, cy, text, ha="center", va="center", color=INK)
    return (cx, cy - h / 2 - PAD), (cx, cy + h / 2 + PAD)


def _arrow(ax, start, end, colour=INK, dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=6, color=colour,
        linewidth=0.7, linestyle="--" if dashed else "-", shrinkA=1, shrinkB=1,
    ))


def _draw_hierarchy(ax):
    """Panel (a): the four dependent fields and the 17 states they admit."""
    ax.set_title("(a) Task hierarchy", loc="left", fontsize=8.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ps_bottom, _ = _box(ax, 5.0, 8.9, 4.0, 0.9, "promise_status\nYes / No")

    _, no_top = _box(ax, 1.8, 6.6, 3.0, 0.9, "VT = ES = EQ\n= N/A")
    yes_bottom, yes_top = _box(
        ax, 6.9, 6.6, 5.2, 0.9,
        "verification_timeline (4)\nevidence_status (Yes / No)",
    )
    _arrow(ax, ps_bottom, no_top)
    _arrow(ax, ps_bottom, yes_top)
    ax.text(2.9, 7.95, "No", ha="center", color=GREY)
    ax.text(6.4, 7.95, "Yes", ha="center", color=GREY)

    _, esno_top = _box(ax, 5.7, 4.2, 2.2, 0.8, "EQ = N/A")
    _, esyes_top = _box(ax, 8.5, 4.2, 2.2, 0.8, "EQ (3)")
    _arrow(ax, yes_bottom, esno_top)
    _arrow(ax, yes_bottom, esyes_top)
    ax.text(5.4, 5.35, "ES = No", ha="center", color=GREY)
    ax.text(8.8, 5.35, "ES = Yes", ha="center", color=GREY)

    ax.text(5.0, 2.4,
            f"$1 + 4\\times(1+3) = {len(STATES)}$ legal tuples",
            ha="center", color=INK)
    ax.text(5.0, 1.6, "of $2\\times5\\times3\\times4 = 120$ combinations",
            ha="center", color=INK)
    ax.text(5.0, 0.7, "the other 103 are hierarchy-invalid",
            ha="center", style="italic", color=GREY)


def _draw_routes(ax):
    """Panel (b): alternative decision routes, deliberately not a stack."""
    ax.set_title("(b) Decision routes (alternatives, not stages)",
                 loc="left", fontsize=8.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Column centres 3.2 apart with 2.6-wide boxes leaves 0.24 of clear space
    # once PAD is added on both sides.
    columns = (1.8, 5.0, 8.2)
    width = 2.6

    base_bottom, _ = _box(ax, 5.0, 8.9, 5.6, 0.9,
                          "base probabilities\n(identical for every method)")

    calibrations = (("no\ncalibration", "none"), ("global\nbias", "global"),
                    ("conditional\nbias", "conditional"))
    calib_bottoms = {}
    for cx, (label, key) in zip(columns, calibrations):
        bottom, top = _box(ax, cx, 6.3, width, 0.9, label,
                           edgecolor=ROUTE_COLOURS[key])
        _arrow(ax, base_bottom, top, colour=ROUTE_COLOURS[key])
        calib_bottoms[key] = bottom

    outputs = (("independent\nargmax", "M0", "may be invalid"),
               ("projection", "M1 / M2 / M3", "always valid"),
               ("17-state\ndecoding", "M4 / M5 / M6", "always valid"))
    output_tops = {}
    for cx, (label, methods, validity) in zip(columns, outputs):
        _, top = _box(ax, cx, 3.3, width, 0.9, label)
        output_tops[methods] = top
        ax.text(cx, 2.3, methods, ha="center", weight="bold", color=INK)
        ax.text(cx, 1.6, validity, ha="center", style="italic", color=GREY)

    # Calibration and output rule are chosen independently, so each calibration
    # feeds both structured output rules. Dashed, because any one method takes
    # exactly one of these routes -- the arrows are alternatives, not a flow.
    for key in ("none", "global", "conditional"):
        for methods in ("M1 / M2 / M3", "M4 / M5 / M6"):
            _arrow(ax, calib_bottoms[key], output_tops[methods],
                   colour=GREY, dashed=True)
    _arrow(ax, calib_bottoms["none"], output_tops["M0"],
           colour=ROUTE_COLOURS["none"])


def build(out_path) -> Path:
    """Render the figure and return the path written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.0, 1.15]},
    )
    _draw_hierarchy(left)
    _draw_routes(right)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    from paper.data import REPO_ROOT

    print(f"wrote {build(REPO_ROOT / 'figures' / 'figure1_hierarchy.pdf')}")


if __name__ == "__main__":
    main()
