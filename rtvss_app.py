"""
REAL-TIME VIDEO STREAMING SCHEDULER (RTVSS)
OS Coursework — Python Desktop App

HOW TO RUN:
  pip install matplotlib numpy
  python rtvss_app.py   (or F5 in VS Code)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading, queue, random
from collections import deque

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import numpy as np
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable,"-m","pip","install","matplotlib","numpy"])
    import matplotlib; matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import numpy as np

# ── Colours ──────────────────────────────────────────────────────────────────
BG     = "#0f1923"
PANEL  = "#16212e"
BORDER = "#1e3048"
WHITE  = "#e8f0f8"
GREY   = "#5a7a96"
BLUE   = "#00c8ff"
RED    = "#ff4560"
GREEN  = "#00e676"
YELLOW = "#ffab00"
PURPLE = "#b388ff"
STREAM_COLORS = [BLUE, RED, GREEN, YELLOW, PURPLE, "#ff6d00", "#00bfa5"]

HISTORY = 60          # number of ticks shown on charts
ABR_BW  = [0.5, 1.5, 3.0, 6.0, 20.0]   # Mbps quality ladder

# ═════════════════════════════════════════════════════════════════════════════
#   SCHEDULING ALGORITHMS
# ═════════════════════════════════════════════════════════════════════════════

def round_robin(tasks, quantum):
    """
    Round Robin — OS fair time-sharing.
    Each task gets `quantum` ms of CPU. If not done, goes back to end of queue.
    Prevents any one stream from hogging the CPU.
    """
    q      = [dict(t) for t in tasks]
    result = []
    clock  = 0.0
    guard  = 0
    while q and guard < 500:
        t   = q.pop(0)
        run = min(t["burst"], quantum)
        result.append({
            "name":     t["name"],
            "color":    t["color"],
            "start":    clock,
            "duration": run,
            "finish":   clock + run,
            "missed":   (clock + run) > t["deadline"],
        })
        clock       += run
        t["burst"]  -= run
        if t["burst"] > 0.01:
            q.append(t)          # preempt — put back at end
        guard += 1
    return result


def priority_schedule(tasks):
    """
    Priority Scheduling — highest priority task runs first.
    Priority 5 = most important (e.g. live stream).
    Priority 1 = least important (e.g. archived video).
    Risk: low-priority streams may starve.
    """
    result = []
    clock  = 0.0
    for t in sorted(tasks, key=lambda x: -x["priority"]):
        finish = clock + t["burst"]
        result.append({
            "name":     t["name"],
            "color":    t["color"],
            "start":    clock,
            "duration": t["burst"],
            "finish":   finish,
            "missed":   finish > t["deadline"],
        })
        clock = finish
    return result


def edf_schedule(tasks):
    """
    Earliest Deadline First (EDF) — optimal real-time scheduler.
    The task with the nearest deadline always runs next.
    Proven: if any algorithm can meet all deadlines, EDF can too.
    Best for live video where frames must display by 33ms (30fps).
    """
    result = []
    clock  = 0.0
    for t in sorted(tasks, key=lambda x: x["deadline"]):   # nearest first
        finish = clock + t["burst"]
        result.append({
            "name":     t["name"],
            "color":    t["color"],
            "start":    clock,
            "duration": t["burst"],
            "finish":   finish,
            "missed":   finish > t["deadline"],
        })
        clock = finish
    return result


# ═════════════════════════════════════════════════════════════════════════════
#   SIMULATION ENGINE  —  background thread
# ═════════════════════════════════════════════════════════════════════════════

class SimEngine(threading.Thread):
    """
    Runs in a background thread. Each tick it:
      1. Allocates bandwidth to streams (resource partitioning)
      2. Creates one frame task per stream (like process creation)
      3. Schedules tasks with chosen algorithm
      4. Calculates latency, drops, buffer levels
      5. Applies ABR (adaptive bitrate — like CPU frequency scaling)
      6. Posts results to a queue for the GUI to read safely
    """

    def __init__(self, streams, algo, quantum, total_bw,
                 result_q, stop_evt, speed_ref):
        super().__init__(daemon=True)
        self.streams   = streams
        self.algo      = algo
        self.quantum   = quantum
        self.total_bw  = total_bw
        self.result_q  = result_q
        self.stop_evt  = stop_evt
        self.speed_ref = speed_ref   # list[float] so it's mutable from GUI
        self.tick = 0
        self.tid  = 0

    def run(self):
        while not self.stop_evt.is_set():
            self.tick += 1
            self._tick()
            speed = max(0.5, self.speed_ref[0])
            self.stop_evt.wait(0.8 / speed)

    def _tick(self):
        S = self.streams
        if not S:
            return

        # 1. Bandwidth allocation — fair share per stream
        total_req = sum(s["bw"] for s in S) or 1
        cap       = float(self.total_bw)
        for s in S:
            s["alloc_bw"] = (s["bw"] / total_req) * min(total_req, cap)

        # 2. Create frame tasks (one per stream)
        tasks = []
        for s in S:
            self.tid += 1
            burst    = max(1.0, s["bw"] * 2.0 + random.gauss(0, 1.5))
            deadline = (33.0 if s["deadline"] == "Strict" else 100.0) + random.uniform(0, 5)
            tasks.append({
                "id":       self.tid,
                "name":     s["name"],
                "color":    s["color"],
                "priority": s["priority"],
                "burst":    burst,
                "deadline": deadline,
            })

        # 3. Schedule
        if   self.algo == "Round Robin": sched = round_robin(tasks, self.quantum)
        elif self.algo == "Priority":    sched = priority_schedule(tasks)
        else:                            sched = edf_schedule(tasks)

        # 4. Calculate outcomes
        log    = []
        drops  = 0

        for s in S:
            seg = next((x for x in sched if x["name"] == s["name"]), None)
            if not seg:
                continue

            stress       = max(0.0, 1.0 - s["alloc_bw"] / max(s["bw"], 0.01))
            s["latency"] = seg["finish"] + abs(random.gauss(0, 3)) + stress * 20

            dl     = 33 if s["deadline"] == "Strict" else 100
            missed = s["latency"] > dl and s["deadline"] == "Strict"
            bw_drop= stress > 0.35 and random.random() < stress * 0.5

            if missed or bw_drop:
                s["dropped"] += 1
                drops        += 1
                s["buffer"]   = max(0.0, s["buffer"] - 8.0)
                why = "deadline" if missed else "low-BW"
                log.append(("DROP",
                    f"DROPPED  {s['name']}  reason={why}  "
                    f"lat={s['latency']:.0f}ms  dl={dl}ms"))
            else:
                s["delivered"] += 1
                fill            = 4.0 - stress * 3.0 + random.uniform(0, 2)
                s["buffer"]     = min(100.0, s["buffer"] + fill)

            # 5. ABR feedback — adjust quality based on buffer
            if   s["buffer"] < 20 and s["abr"] > 0:
                s["abr"] -= 1
                s["bw"]   = ABR_BW[s["abr"]]
                log.append(("ABR", f"ABR DOWN  {s['name']} -> {s['bw']}Mbps  buf={s['buffer']:.0f}%"))
            elif s["buffer"] > 80 and s["abr"] < len(ABR_BW) - 1:
                s["abr"] += 1
                s["bw"]   = ABR_BW[s["abr"]]
                log.append(("ABR", f"ABR UP    {s['name']} -> {s['bw']}Mbps  buf={s['buffer']:.0f}%"))

        avg_cpu = min(99.0, sum(s["bw"] for s in S) / max(cap, 1) * 85 + random.uniform(0, 10))
        avg_lat = sum(s["latency"] for s in S) / len(S)
        used_bw = sum(s["alloc_bw"] for s in S)

        top3 = ", ".join(f"{t['name']}({t['burst']:.0f}ms)" for t in tasks[:3])
        log.append(("SCHED", f"[{self.algo}] tick={self.tick}  {top3}"))

        self.result_q.put({
            "tick":       self.tick,
            "avg_cpu":    avg_cpu,
            "avg_lat":    avg_lat,
            "used_bw":    used_bw,
            "tick_drops": drops,
            "scheduled":  sched,
            "log":        log,
            "streams":    [dict(s) for s in S],
        })


# ═════════════════════════════════════════════════════════════════════════════
#   GUI APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

class App:

    RES_MAP = {
        "480p  (1.5 Mbps)":  1.5,
        "720p  (3 Mbps)":    3.0,
        "1080p (6 Mbps)":    6.0,
        "4K    (20 Mbps)":  20.0,
    }

    def __init__(self, root):
        self.root = root
        root.title("Real-Time Video Streaming Scheduler")
        root.geometry("1300x760")
        root.configure(bg=BG)
        root.resizable(True, True)

        # state
        self.streams    = []
        self.sid        = 0
        self.running    = False
        self.stop_evt   = threading.Event()
        self.result_q   = queue.Queue()
        self.sim        = None
        self.speed_ref  = [1.0]

        # history buffers — always HISTORY length, so set_ydata never mismatches
        self.h_cpu  = deque([0.0] * HISTORY, maxlen=HISTORY)
        self.h_lat  = deque([0.0] * HISTORY, maxlen=HISTORY)
        self.h_drop = deque([0.0] * HISTORY, maxlen=HISTORY)
        self.h_bw   = deque([0.0] * HISTORY, maxlen=HISTORY)

        self.tot_drop = 0
        self.tot_del  = 0
        self.tick     = 0
        self.log_all  = []

        self._styles()
        self._build_ui()
        self._add_defaults()
        self._poll()           # start 50ms GUI refresh loop

    # ─── STYLES ──────────────────────────────────────────────────────────────

    def _styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TNotebook",     background=PANEL, borderwidth=0)
        s.configure("TNotebook.Tab", background=PANEL, foreground=GREY,
                    padding=[12, 7],  font=("Consolas", 9, "bold"))
        s.map("TNotebook.Tab",
              background=[("selected", BG)],
              foreground=[("selected", BLUE)])
        s.configure("TCombobox", fieldbackground=BG, background=BG,
                    foreground=WHITE, font=("Consolas", 9))
        s.map("TCombobox",
              fieldbackground=[("readonly", BG)],
              foreground=[("readonly", WHITE)])
        s.configure("Treeview", background=PANEL, foreground=WHITE,
                    fieldbackground=PANEL, rowheight=24,
                    font=("Consolas", 9))
        s.configure("Treeview.Heading", background=BG, foreground=GREY,
                    font=("Consolas", 9, "bold"))
        s.map("Treeview", background=[("selected", BORDER)])

    # ─── BUILD UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Two columns: sidebar | main
        left = tk.Frame(self.root, bg=PANEL, width=248)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        right = tk.Frame(self.root, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._sidebar(left)
        self._main_area(right)

    # ─── SIDEBAR ─────────────────────────────────────────────────────────────

    def _sidebar(self, L):
        tk.Label(L, text="RTVSS", font=("Consolas", 15, "bold"),
                 bg=PANEL, fg=BLUE).pack(pady=(14, 0))
        tk.Label(L, text="Video Streaming Scheduler",
                 font=("Consolas", 8), bg=PANEL, fg=GREY).pack(pady=(0, 4))
        self._sep(L)

        # ── Controls ─────────────────────────────────────────────────
        self._hdr(L, "CONTROLS")
        row = tk.Frame(L, bg=PANEL)
        row.pack(fill=tk.X, padx=10, pady=4)

        self.btn_start = self._mk_btn(row, "▶  START", BLUE, "#000", self.start)
        self.btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        self.btn_stop = self._mk_btn(row, "■  STOP", RED, "#fff", self.stop)
        self.btn_stop.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_stop.config(state=tk.DISABLED)

        self._lbl(L, "Speed")
        self.speed_scale = tk.Scale(
            L, from_=0.5, to=4, resolution=0.5, orient=tk.HORIZONTAL,
            bg=PANEL, fg=WHITE, troughcolor=BG, highlightthickness=0,
            font=("Consolas", 8), activebackground=BLUE,
            command=lambda v: self.speed_ref.__setitem__(0, float(v)))
        self.speed_scale.set(1)
        self.speed_scale.pack(fill=tk.X, padx=10)
        self._sep(L)

        # ── Algorithm ────────────────────────────────────────────────
        self._hdr(L, "ALGORITHM")
        self._lbl(L, "Select Algorithm")
        self.algo_var = tk.StringVar(value="EDF")
        algo_cb = ttk.Combobox(L, textvariable=self.algo_var, state="readonly",
                               values=["Round Robin", "Priority", "EDF"],
                               font=("Consolas", 9))
        algo_cb.pack(fill=tk.X, padx=10, pady=4)
        algo_cb.bind("<<ComboboxSelected>>", self._algo_changed)

        # Quantum row — only shown when Round Robin selected
        self.q_frame = tk.Frame(L, bg=PANEL)
        self._lbl(self.q_frame, "Quantum (ms)")
        self.quantum_var = tk.StringVar(value="16")
        tk.Entry(self.q_frame, textvariable=self.quantum_var,
                 bg=BG, fg=WHITE, insertbackground=BLUE,
                 font=("Consolas", 9), relief=tk.FLAT,
                 highlightbackground=BORDER, highlightthickness=1
                 ).pack(fill=tk.X, pady=2)
        # hidden by default (EDF selected)
        self._sep(L)

        # ── Settings ─────────────────────────────────────────────────
        self._hdr(L, "SETTINGS")
        self._lbl(L, "Total Bandwidth (Mbps)")
        self.bw_var = tk.StringVar(value="50")
        tk.Entry(L, textvariable=self.bw_var, bg=BG, fg=WHITE,
                 insertbackground=BLUE, font=("Consolas", 9),
                 relief=tk.FLAT, highlightbackground=BORDER,
                 highlightthickness=1).pack(fill=tk.X, padx=10, pady=2)
        self._sep(L)

        # ── Add Stream ───────────────────────────────────────────────
        self._hdr(L, "ADD STREAM")
        self._lbl(L, "Name")
        self.nm_var = tk.StringVar()
        tk.Entry(L, textvariable=self.nm_var, bg=BG, fg=WHITE,
                 insertbackground=BLUE, font=("Consolas", 9),
                 relief=tk.FLAT, highlightbackground=BORDER,
                 highlightthickness=1).pack(fill=tk.X, padx=10, pady=2)

        self._lbl(L, "Resolution")
        self.res_var = tk.StringVar(value="720p  (3 Mbps)")
        ttk.Combobox(L, textvariable=self.res_var, state="readonly",
                     values=list(self.RES_MAP.keys()),
                     font=("Consolas", 9)).pack(fill=tk.X, padx=10, pady=2)

        self._lbl(L, "Priority  (1=low  5=high)")
        self.pri_var = tk.StringVar(value="3")
        ttk.Combobox(L, textvariable=self.pri_var, state="readonly",
                     values=["1", "2", "3", "4", "5"],
                     font=("Consolas", 9)).pack(fill=tk.X, padx=10, pady=2)

        self._lbl(L, "Deadline")
        self.dl_var = tk.StringVar(value="Strict")
        ttk.Combobox(L, textvariable=self.dl_var, state="readonly",
                     values=["Strict", "Soft"],
                     font=("Consolas", 9)).pack(fill=tk.X, padx=10, pady=2)

        self._mk_btn(L, "+  Add Stream", BORDER, BLUE,
                     self.add_stream).pack(fill=tk.X, padx=10, pady=8)

    # ─── MAIN AREA ───────────────────────────────────────────────────────────

    def _main_area(self, R):
        # Status bar
        bar = tk.Frame(R, bg=PANEL, height=36)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        self.lbl_status = tk.Label(bar, text="●  IDLE",
                                   font=("Consolas", 9, "bold"),
                                   bg=PANEL, fg=GREY)
        self.lbl_status.pack(side=tk.LEFT, padx=14, pady=8)

        self.lbl_tick = tk.Label(bar, text="Tick: 0",
                                 font=("Consolas", 9), bg=PANEL, fg=GREY)
        self.lbl_tick.pack(side=tk.LEFT, padx=8)

        self.lbl_algo_badge = tk.Label(bar, text="[EDF]",
                                       font=("Consolas", 9, "bold"),
                                       bg=PANEL, fg=BLUE)
        self.lbl_algo_badge.pack(side=tk.LEFT, padx=4)

        # Stat cards
        cards = tk.Frame(R, bg=BG)
        cards.pack(fill=tk.X, padx=8, pady=8)
        self.v_cpu  = self._card(cards, "CPU %",     "0",  BLUE)
        self.v_bw   = self._card(cards, "BW Mbps",   "0",  YELLOW)
        self.v_lat  = self._card(cards, "Latency ms","0",  PURPLE)
        self.v_drop = self._card(cards, "Dropped",   "0",  RED)
        self.v_del  = self._card(cards, "Delivered", "0",  GREEN)

        # Tabs
        nb = ttk.Notebook(R)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self._tab_charts(nb)
        self._tab_gantt(nb)
        self._tab_streams(nb)
        self._tab_log(nb)

    # ─── TAB: LIVE CHARTS ────────────────────────────────────────────────────

    def _tab_charts(self, nb):
        tab = tk.Frame(nb, bg=BG)
        nb.add(tab, text="  📊 Live Charts  ")

        # Single figure, 2×2 grid
        self.fig_ch, axes = plt.subplots(2, 2, figsize=(9, 4.5),
                                          facecolor=PANEL)
        self.fig_ch.subplots_adjust(hspace=0.52, wspace=0.32,
                                     left=0.08, right=0.97,
                                     top=0.92, bottom=0.10)

        self.ax_cpu,  self.ax_lat  = axes[0]
        self.ax_drop, self.ax_bw   = axes[1]

        X = list(range(HISTORY))

        for ax, title, color in [
            (self.ax_cpu,  "CPU Usage (%)",     BLUE),
            (self.ax_lat,  "Latency (ms)",       PURPLE),
            (self.ax_drop, "Dropped Frames",     RED),
            (self.ax_bw,   "Bandwidth (Mbps)",   YELLOW),
        ]:
            ax.set_facecolor(BG)
            ax.tick_params(colors=GREY, labelsize=7)
            ax.set_title(title, color=WHITE, fontsize=8, pad=4)
            ax.grid(True, color=BORDER, lw=0.4, alpha=0.8)
            for sp in ax.spines.values():
                sp.set_color(BORDER)

        # Create lines once — we only call set_ydata() to update them
        # This avoids re-creating axes and is the correct matplotlib pattern
        self.ln_cpu,  = self.ax_cpu.plot(X, [0.0] * HISTORY,
                                          color=BLUE,   lw=1.5)
        self.ln_lat,  = self.ax_lat.plot(X, [0.0] * HISTORY,
                                          color=PURPLE, lw=1.5)
        self.ln_bw,   = self.ax_bw.plot(X,  [0.0] * HISTORY,
                                          color=YELLOW, lw=1.5)

        # Static deadline line on latency chart
        self.ax_lat.axhline(33, color=RED, lw=0.8, ls="--", alpha=0.8)
        self.ax_lat.text(1, 36, "33ms limit", color=RED, fontsize=7)
        self.ax_lat.set_ylim(0, 150)
        self.ax_cpu.set_ylim(0, 110)

        # Store fill polygon handles so we can remove them properly
        self._fill_cpu = None
        self._fill_lat = None

        self.cv_ch = FigureCanvasTkAgg(self.fig_ch, tab)
        self.cv_ch.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                         padx=6, pady=6)

    # ─── TAB: GANTT ──────────────────────────────────────────────────────────

    def _tab_gantt(self, nb):
        tab = tk.Frame(nb, bg=BG)
        nb.add(tab, text="  ⚙ Gantt Chart  ")

        tk.Label(tab,
                 text="CPU Scheduling — Gantt Chart  "
                      "(red border = missed deadline)",
                 font=("Consolas", 9), bg=BG, fg=GREY
                 ).pack(anchor=tk.W, padx=10, pady=(8, 2))

        self.fig_g = plt.Figure(figsize=(9, 3.8), facecolor=PANEL)
        self.fig_g.subplots_adjust(left=0.15, right=0.97,
                                    top=0.88, bottom=0.18)
        self.ax_g = self.fig_g.add_subplot(111)
        self.ax_g.set_facecolor(BG)
        for sp in self.ax_g.spines.values():
            sp.set_color(BORDER)
        self.ax_g.tick_params(colors=GREY, labelsize=7)
        self.ax_g.set_xlabel("Time (ms)", color=GREY, fontsize=8)
        self.ax_g.text(0.5, 0.5, "Press START to begin",
                       transform=self.ax_g.transAxes,
                       color=GREY, fontsize=11, ha="center", va="center")

        self.cv_g = FigureCanvasTkAgg(self.fig_g, tab)
        self.cv_g.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                        padx=6, pady=6)

    # ─── TAB: STREAMS ────────────────────────────────────────────────────────

    def _tab_streams(self, nb):
        tab = tk.Frame(nb, bg=BG)
        nb.add(tab, text="  📹 Streams  ")

        cols = ("Name", "Res", "Priority", "Deadline",
                "BW Req", "BW Alloc", "Buffer %",
                "Delivered", "Dropped", "Latency ms")
        self.tree = ttk.Treeview(tab, columns=cols, show="headings")
        for col, w in zip(cols, [110, 65, 60, 65, 62, 62, 60, 72, 62, 78]):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor=tk.CENTER)

        sb = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH,
                       expand=True, padx=(8, 0), pady=8)
        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=8, padx=(0, 8))

        self._mk_btn(tab, "✕  Remove Selected",
                     RED, "#fff", self._remove_stream
                     ).pack(pady=(0, 8))

    # ─── TAB: LOG ────────────────────────────────────────────────────────────

    def _tab_log(self, nb):
        tab = tk.Frame(nb, bg=BG)
        nb.add(tab, text="  📋 Event Log  ")

        bar = tk.Frame(tab, bg=BG)
        bar.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(bar, text="Filter:", bg=BG, fg=GREY,
                 font=("Consolas", 9)).pack(side=tk.LEFT)
        self.flt_var = tk.StringVar(value="ALL")
        ttk.Combobox(bar, textvariable=self.flt_var, state="readonly",
                     width=10, values=["ALL", "SCHED", "DROP", "ABR"],
                     font=("Consolas", 9)).pack(side=tk.LEFT, padx=6)
        self.flt_var.trace_add("write", lambda *_: self._redraw_log())

        self._mk_btn(bar, "Clear", PANEL, GREY,
                     self._clear_log).pack(side=tk.LEFT)
        self.lbl_cnt = tk.Label(bar, text="0 events",
                                bg=BG, fg=GREY, font=("Consolas", 9))
        self.lbl_cnt.pack(side=tk.RIGHT, padx=8)

        self.log_box = tk.Text(tab, bg=PANEL, fg=WHITE,
                               font=("Consolas", 9), relief=tk.FLAT,
                               state=tk.DISABLED, wrap=tk.NONE)
        vsb = ttk.Scrollbar(tab, orient=tk.VERTICAL,
                             command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=vsb.set)
        self.log_box.pack(side=tk.LEFT, fill=tk.BOTH,
                          expand=True, padx=(8, 0), pady=(0, 8))
        vsb.pack(side=tk.RIGHT, fill=tk.Y, pady=(0, 8), padx=(0, 8))

        for tag, col in [("SCHED", BLUE), ("DROP", RED),
                          ("ABR", YELLOW), ("SYS", GREEN), ("TIME", GREY)]:
            self.log_box.tag_config(tag, foreground=col)

    # ─── WIDGET HELPERS ──────────────────────────────────────────────────────

    def _hdr(self, p, t):
        tk.Label(p, text=t, font=("Consolas", 8, "bold"),
                 bg=PANEL, fg=GREY).pack(anchor=tk.W, padx=10, pady=(8, 1))

    def _lbl(self, p, t):
        tk.Label(p, text=t, font=("Consolas", 8),
                 bg=PANEL, fg=GREY).pack(anchor=tk.W, padx=10, pady=(4, 0))

    def _sep(self, p):
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, padx=10, pady=6)

    def _mk_btn(self, p, t, bg, fg, cmd):
        return tk.Button(p, text=t, bg=bg, fg=fg,
                         font=("Consolas", 9, "bold"), relief=tk.FLAT,
                         cursor="hand2", activebackground=bg,
                         activeforeground=fg, command=cmd,
                         padx=8, pady=5)

    def _card(self, p, label, val, color):
        f = tk.Frame(p, bg=PANEL, padx=12, pady=8)
        f.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Frame(f, bg=color, height=2).pack(fill=tk.X, pady=(0, 4))
        tk.Label(f, text=label, font=("Consolas", 8),
                 bg=PANEL, fg=GREY).pack()
        v = tk.Label(f, text=val, font=("Consolas", 17, "bold"),
                     bg=PANEL, fg=color)
        v.pack()
        return v

    # ─── STREAM MANAGEMENT ───────────────────────────────────────────────────

    def add_stream(self):
        self.sid += 1
        name  = self.nm_var.get().strip() or f"Stream_{self.sid}"
        bw    = self.RES_MAP.get(self.res_var.get(), 3.0)
        res   = self.res_var.get().split()[0]
        pri   = int(self.pri_var.get())
        dl    = self.dl_var.get()
        color = STREAM_COLORS[(self.sid - 1) % len(STREAM_COLORS)]

        self.streams.append({
            "sid": self.sid, "name": name, "res": res,
            "bw": bw, "priority": pri, "deadline": dl,
            "color": color, "abr": 2, "buffer": 50.0,
            "alloc_bw": 0.0, "latency": 0.0,
            "dropped": 0, "delivered": 0,
        })
        self.nm_var.set("")
        self._refresh_tree()
        self._log("SYS", f'Added "{name}" [{res}, P{pri}, {dl}, {bw}Mbps]')

    def _add_defaults(self):
        for nm, res, pri, dl in [
            ("Live_HD",   "1080p (6 Mbps)",   5, "Strict"),
            ("News_720",  "720p  (3 Mbps)",   4, "Strict"),
            ("VOD_480",   "480p  (1.5 Mbps)", 2, "Soft"),
        ]:
            self.nm_var.set(nm)
            self.res_var.set(res)
            self.pri_var.set(str(pri))
            self.dl_var.set(dl)
            self.add_stream()
        self.nm_var.set("")

    def _remove_stream(self):
        sel = self.tree.selection()
        if not sel:
            return
        name = self.tree.item(sel[0])["values"][0]
        self.streams = [s for s in self.streams if s["name"] != name]
        self.tree.delete(sel[0])
        self._log("SYS", f'Removed "{name}"')

    def _refresh_tree(self):
        for r in self.tree.get_children():
            self.tree.delete(r)
        for s in self.streams:
            self.tree.insert("", tk.END, values=(
                s["name"], s["res"], s["priority"], s["deadline"],
                f"{s['bw']:.1f}", f"{s['alloc_bw']:.1f}",
                f"{s['buffer']:.0f}%",
                s["delivered"], s["dropped"],
                f"{s['latency']:.0f}",
            ))

    # ─── SIMULATION ──────────────────────────────────────────────────────────

    def start(self):
        if self.running:
            return
        if not self.streams:
            messagebox.showwarning("No Streams",
                                   "Add at least one stream first.")
            return
        self.running = True
        self.stop_evt.clear()
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.lbl_status.config(text="●  RUNNING", fg=GREEN)
        algo = self.algo_var.get()
        self.lbl_algo_badge.config(text=f"[{algo}]")

        self.sim = SimEngine(
            streams   = self.streams,
            algo      = algo,
            quantum   = float(self.quantum_var.get() or 16),
            total_bw  = float(self.bw_var.get() or 50),
            result_q  = self.result_q,
            stop_evt  = self.stop_evt,
            speed_ref = self.speed_ref,
        )
        self.sim.start()
        self._log("SYS",
                  f"Started | algo={algo} | "
                  f"streams={len(self.streams)} | "
                  f"bw={self.bw_var.get()}Mbps")

    def stop(self):
        self.running = False
        self.stop_evt.set()
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.lbl_status.config(text="●  STOPPED", fg=RED)
        self._log("SYS",
                  f"Stopped | delivered={self.tot_del} "
                  f"| dropped={self.tot_drop}")

    def _algo_changed(self, _=None):
        algo = self.algo_var.get()
        if algo == "Round Robin":
            self.q_frame.pack(fill=tk.X, padx=10, pady=2)
        else:
            self.q_frame.pack_forget()
        self.lbl_algo_badge.config(text=f"[{algo}]")

    # ─── POLL LOOP (runs on main thread every 50ms) ───────────────────────────

    def _poll(self):
        """
        Drains the result queue and updates all UI elements.
        MUST run on the main thread — tkinter is not thread-safe.
        We never call draw() or widget updates from the SimEngine thread.
        """
        try:
            while True:
                data = self.result_q.get_nowait()
                self._apply(data)
        except queue.Empty:
            pass
        self.root.after(50, self._poll)   # schedule next poll

    def _apply(self, data):
        self.tick = data["tick"]

        # Update history buffers
        self.h_cpu.append(data["avg_cpu"])
        self.h_lat.append(data["avg_lat"])
        self.h_drop.append(float(data["tick_drops"]))
        self.h_bw.append(data["used_bw"])
        self.tot_drop += data["tick_drops"]
        self.tot_del  += max(0, len(data["streams"]) - data["tick_drops"])

        # Stat cards
        self.v_cpu.config( text=f"{data['avg_cpu']:.0f}")
        self.v_bw.config(  text=f"{data['used_bw']:.1f}")
        self.v_lat.config( text=f"{data['avg_lat']:.0f}")
        self.v_drop.config(text=str(self.tot_drop))
        self.v_del.config( text=str(self.tot_del))
        self.lbl_tick.config(text=f"Tick: {self.tick}")

        # Log lines
        for tag, msg in data["log"]:
            self._log(tag, msg)

        # Charts
        self._update_charts()
        self._update_gantt(data["scheduled"])

        # Sync stream snapshots back
        for snap in data["streams"]:
            s = next((x for x in self.streams
                      if x["name"] == snap["name"]), None)
            if s:
                s.update(snap)
        self._refresh_tree()

    # ─── CHART UPDATES ───────────────────────────────────────────────────────

    def _update_charts(self):
        X      = list(range(HISTORY))
        y_cpu  = list(self.h_cpu)
        y_lat  = list(self.h_lat)
        y_bw   = list(self.h_bw)
        y_drop = list(self.h_drop)

        # ── Line charts: just update ydata — no cla(), no re-create ──
        self.ln_cpu.set_ydata(y_cpu)
        self.ln_lat.set_ydata(y_lat)
        self.ln_bw.set_ydata(y_bw)

        # Auto-scale y axes
        self.ax_cpu.set_ylim(0, max(110, max(y_cpu)  * 1.1))
        self.ax_bw.set_ylim( 0, max(5,   max(y_bw)   * 1.2))
        self.ax_lat.set_ylim(0, max(150, max(y_lat)  * 1.1))

        # ── Fill under CPU line ────────────────────────────────────────
        # Remove old fill first (correct way for new matplotlib)
        if self._fill_cpu is not None:
            self._fill_cpu.remove()
        self._fill_cpu = self.ax_cpu.fill_between(
            X, y_cpu, alpha=0.12, color=BLUE)

        if self._fill_lat is not None:
            self._fill_lat.remove()
        self._fill_lat = self.ax_lat.fill_between(
            X, y_lat, alpha=0.10, color=PURPLE)

        # ── Drop bars: cla() is fine here since it's a bar chart ──────
        self.ax_drop.cla()
        self.ax_drop.set_facecolor(BG)
        self.ax_drop.tick_params(colors=GREY, labelsize=7)
        self.ax_drop.set_title("Dropped Frames", color=WHITE,
                                fontsize=8, pad=4)
        self.ax_drop.grid(True, color=BORDER, lw=0.4, alpha=0.8)
        for sp in self.ax_drop.spines.values():
            sp.set_color(BORDER)
        colors = [RED if v > 0 else BORDER for v in y_drop]
        self.ax_drop.bar(X, y_drop, color=colors, alpha=0.9)
        self.ax_drop.set_ylim(0, max(3, max(y_drop) + 1))

        # Draw once per poll cycle
        self.cv_ch.draw_idle()

    def _update_gantt(self, scheduled):
        ax = self.ax_g
        ax.cla()
        ax.set_facecolor(BG)
        ax.tick_params(colors=GREY, labelsize=7)
        ax.set_xlabel("Time (ms)", color=GREY, fontsize=8)
        ax.set_title(
            f"Gantt — {self.algo_var.get()}   "
            f"(tick {self.tick})   "
            f"red border = missed deadline",
            color=WHITE, fontsize=8, pad=4)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.grid(True, axis="x", color=BORDER, lw=0.4, alpha=0.6)

        if not scheduled:
            ax.text(0.5, 0.5, "No tasks scheduled",
                    transform=ax.transAxes,
                    color=GREY, fontsize=10, ha="center", va="center")
            self.cv_g.draw_idle()
            return

        # One row per stream name
        names = list(dict.fromkeys(s["name"] for s in scheduled))
        ymap  = {n: i for i, n in enumerate(names)}

        for seg in scheduled:
            y = ymap[seg["name"]]
            # Coloured bar for the task
            ax.barh(y, seg["duration"], left=seg["start"],
                    color=seg["color"] + "cc",
                    edgecolor=seg["color"],
                    linewidth=0.8, height=0.55)
            # Duration label inside bar
            if seg["duration"] > 2:
                ax.text(seg["start"] + seg["duration"] / 2, y,
                        f"{seg['duration']:.0f}ms",
                        ha="center", va="center",
                        color="white", fontsize=7, fontweight="bold")
            # Red outline if deadline missed
            if seg["missed"]:
                ax.barh(y, seg["duration"], left=seg["start"],
                        color="none", edgecolor=RED,
                        linewidth=2.5, height=0.55)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, color=WHITE, fontsize=8)
        ax.invert_yaxis()
        self.cv_g.draw_idle()

    # ─── LOG ─────────────────────────────────────────────────────────────────

    def _log(self, tag, msg):
        self.log_all.append((tag, msg))
        if hasattr(self, "flt_var"):
            if self.flt_var.get() != "ALL" and tag != self.flt_var.get():
                return
        self.log_box.config(state=tk.NORMAL)
        self.log_box.insert(tk.END, f"t={self.tick:04d}  ", "TIME")
        self.log_box.insert(tk.END, f"[{tag}]  ", tag)
        self.log_box.insert(tk.END, f"{msg}\n")
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)
        self.lbl_cnt.config(text=f"{len(self.log_all)} events")

    def _redraw_log(self):
        flt = self.flt_var.get()
        self.log_box.config(state=tk.NORMAL)
        self.log_box.delete("1.0", tk.END)
        for tag, msg in self.log_all:
            if flt != "ALL" and tag != flt:
                continue
            self.log_box.insert(tk.END, "t=----  ", "TIME")
            self.log_box.insert(tk.END, f"[{tag}]  ", tag)
            self.log_box.insert(tk.END, f"{msg}\n")
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)

    def _clear_log(self):
        self.log_all.clear()
        self.log_box.config(state=tk.NORMAL)
        self.log_box.delete("1.0", tk.END)
        self.log_box.config(state=tk.DISABLED)
        self.lbl_cnt.config(text="0 events")


# ═════════════════════════════════════════════════════════════════════════════
#   RUN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app  = App(root)
    root.protocol("WM_DELETE_WINDOW",
                  lambda: (app.stop(), root.destroy()))
    root.mainloop()