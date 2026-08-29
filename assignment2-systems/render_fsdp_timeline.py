"""Render a Gantt-style timeline from the FSDP nsys SQLite export.

Shows, per rank: the NVTX phase bands (forward/backward/grad_sync/optimizer)
in the background and the CUDA kernels on the GPU on top — NCCL kernels
(weight all-gather / gradient reduce-scatter) in red, everything else in gray.

The key visual for the writeup: all NCCL all-gather kernels sit *inside* the
forward band (each layer gathers its weight right before computing), and the
gradient reduce-scatters sit inside the backward band, so communication never
adds a separate serial phase.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_CJK = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
if Path(_CJK).exists():
    fm.fontManager.addfont(_CJK)
    matplotlib.rcParams["font.family"] = [fm.FontProperties(fname=_CJK).get_name(), "DejaVu Sans"]

PHASES = ("forward", "backward", "grad_sync", "optimizer")
PHASE_COLORS = {"forward": "#9ecae1", "backward": "#fdae6b", "grad_sync": "#d8b7f4", "optimizer": "#a1d99b"}
NS = 1e6  # ns -> ms


def load(path: Path):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT globalPid, pid FROM PROCESSES WHERE name LIKE '%python%'")
    pyprocs = cur.fetchall()
    cur.execute(
        "SELECT k.start, k.end, k.globalPid, d.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds d ON k.demangledName=d.id"
    )
    kernels = cur.fetchall()
    nvtx = {}
    for phase in PHASES:
        cur.execute("SELECT start, end, globalTid FROM NVTX_EVENTS WHERE text=?", (phase,))
        nvtx[phase] = cur.fetchall()
    return pyprocs, kernels, nvtx


def rank_phases_kernels(pyprocs, kernels, nvtx, gp_pid):
    gp, pid = gp_pid
    phases = []
    for phase in PHASES:
        for s, e, tid in nvtx[phase]:
            if tid - pid == gp:
                phases.append((s / NS, e / NS, phase))
    phases.sort()
    klist = []
    for s, e, kgp, name in kernels:
        if kgp == gp:
            klist.append((s / NS, (e - s) / NS, "nccl" in name.lower()))
    return phases, klist


def rank_processes(pyprocs, kernels, nvtx):
    kernel_pids = {kgp for _, _, kgp, _ in kernels}
    return [t for t in pyprocs if t[0] in kernel_pids]


def draw_lane(ax, phases, klist, y0, y1):
    ax.axhspan(y0, y1, color="#f7f7f7", zorder=0)
    for s, e, phase in phases:
        ax.add_patch(plt.Rectangle((s, y0 + 0.08), e - s, y1 - y0 - 0.16,
                                   facecolor=PHASE_COLORS[phase], alpha=0.55, edgecolor="none", zorder=1))
    compute = [(s, w) for s, w, c in klist if not c and w >= 0.003]
    if compute:
        ax.broken_barh(compute, (y0 + 0.18, (y1 - y0) * 0.28), facecolors="#bdbdbd", edgecolor="none", zorder=2)
    comm = [(s, w) for s, w, c in klist if c]
    if comm:
        ax.broken_barh(comm, (y0 + 0.18, (y1 - y0) * 0.28), facecolors="#d62728", edgecolor="none", zorder=3)


def render(db: Path, title: str, ax):
    pyprocs, kernels, nvtx = load(db)
    ranks = rank_processes(pyprocs, kernels, nvtx)
    # Window: last two forward starts through last optimizer end.
    forwards = []
    last_optimizer_end = 0.0
    for gp, pid in ranks:
        phases, _ = rank_phases_kernels(pyprocs, kernels, nvtx, (gp, pid))
        for s, e, phase in phases:
            if phase == "forward":
                forwards.append(s)
            if phase == "optimizer":
                last_optimizer_end = max(last_optimizer_end, e)
    forwards.sort()
    x0 = (forwards[-2] - 20) if len(forwards) >= 2 else forwards[0] - 20
    x1 = last_optimizer_end + 20

    n = len(ranks)
    for i, (gp, pid) in enumerate(ranks):
        phases, klist = rank_phases_kernels(pyprocs, kernels, nvtx, (gp, pid))
        phases_w = [(s, e, p) for s, e, p in phases if e >= x0 and s <= x1]
        y_top = n - i
        y_bot = n - i - 1
        draw_lane(ax, phases_w, klist, y_bot, y_top)
        ax.text(x0 - 2, (y_top + y_bot) / 2, f"rank {pid}", ha="right", va="center", fontsize=9)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks([])
    ax.set_xlabel("时间 (ms)")
    ax.set_title(title, fontsize=11)


def main() -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    render(Path("profiles/fsdp_xl.sqlite"), "FSDP（xl 模型，2 GPU，权重 all-gather 在 forward 内完成）", ax)
    handles = [
        Patch(facecolor="#9ecae1", label="forward"),
        Patch(facecolor="#fdae6b", label="backward"),
        Patch(facecolor="#d8b7f4", label="grad_sync"),
        Patch(facecolor="#a1d99b", label="optimizer"),
        Patch(facecolor="#d62728", label="NCCL kernel"),
        Patch(facecolor="#bdbdbd", label="compute kernel"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = Path("figures")
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fsdp_timeline.png", dpi=150)
    print("saved figures/fsdp_timeline.png")


if __name__ == "__main__":
    main()
