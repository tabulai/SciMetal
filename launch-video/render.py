#!/usr/bin/env python3
"""Render the SciMetal launch video (1920x1080, 30 fps) with Pillow and ffmpeg.

Every number on screen comes from a real run on an Apple M1 Ultra:
  data/race.json        per-call timestamps from data/hundred.py (100 gaussian_filter calls each)
  data/race_output.txt  stdout of data/race.py, replayed in the terminal scene
  README.md             benchmark table, used by the chart scene

Usage:
  python3 render.py                     # writes scimetal-launch-silent.mp4 (audio.py adds sound)
  python3 render.py --still 3.0 21.5    # writes stills/still-03.00.png, ... for review
"""
import bisect
import json
import os
import random
import re
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS, SS = 1920, 1080, 30, 2  # SS: supersampling factor for antialiased shapes

# Paper and ink, one accent. No gradients, no glow.
BG = (243, 240, 233)
INK = (24, 23, 21)
MUTED = (120, 116, 108)
TRACK = (226, 221, 211)
ACCENT = (214, 72, 30)
GREY = (168, 163, 154)
TERM_BG = (28, 28, 27)
TERM_FG = (228, 226, 220)
TERM_DIM = (126, 124, 118)
TERM_ACCENT = (246, 128, 84)
STR = (46, 118, 70)
NUM = (38, 90, 168)
PUNCT = (112, 108, 100)

SANS = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"


@lru_cache(maxsize=None)
def F(kind, size, weight="Regular"):
    f = ImageFont.truetype(SANS if kind == "sans" else MONO, round(size * SS))
    f.set_variation_by_name(weight)
    return f


def mix(c, a, bg=BG):
    a = max(0.0, min(1.0, a))
    return tuple(round(b + (x - b) * a) for x, b in zip(c, bg))


def ramp(t, start, dur):
    return max(0.0, min(1.0, (t - start) / dur))


def ease_out(x):
    return 1 - (1 - x) ** 3


class Frame:
    def __init__(self, bg=BG):
        self.im = Image.new("RGB", (W * SS, H * SS), bg)
        self.d = ImageDraw.Draw(self.im)

    def text(self, x, y, s, f, fill, anchor="ls"):
        if s:
            self.d.text((x * SS, y * SS), s, font=f, fill=fill, anchor=anchor)

    @staticmethod
    def width(s, f):
        return f.getlength(s) / SS

    def rich(self, x, y, segs, center=False):
        if center:
            x -= sum(self.width(s, f) for s, f, _ in segs) / 2
        for s, f, c in segs:
            self.text(x, y, s, f, c)
            x += self.width(s, f)
        return x

    def box(self, x0, y0, x1, y1, fill, r=0):
        b = [x0 * SS, y0 * SS, x1 * SS, y1 * SS]
        if r:
            self.d.rounded_rectangle(b, radius=r * SS, fill=fill)
        else:
            self.d.rectangle(b, fill=fill)

    def out(self):
        return self.im.resize((W, H), Image.LANCZOS)


# --- typing -------------------------------------------------------------------

def keystrokes(text, start, seed, base=0.07):
    """Human-ish key times: uneven gaps, short hesitations, slower around punctuation."""
    rng = random.Random(seed)
    t, out = start, []
    for ch in text:
        dt = base * rng.uniform(0.6, 1.5)
        if ch in ' ._("':
            dt += rng.uniform(0.0, base)
        if rng.random() < 0.07:
            dt += rng.uniform(0.1, 0.25)
        t += dt
        out.append(t)
    return out


def typed(text, stamps, t):
    return text[: bisect.bisect_right(stamps, t)]


def cursor_on(t, idle_since):
    return int(max(0.0, t - idle_since) / 0.53) % 2 == 0


KEYWORDS = {"import", "as", "with", "from"}
TOKEN = re.compile(r'\s+|"[^"]*"?|\d+(?:\.\d+)?|[A-Za-z_]\w*|.')


def code_line(fr, x, y, s, size):
    """Print-style syntax colouring: keywords in semibold ink, a little colour for literals."""
    reg, bold = F("mono", size), F("mono", size, "Semibold")
    for m in TOKEN.finditer(s):
        tok = m.group()
        if tok in KEYWORDS:
            f, c = bold, INK
        elif tok.startswith('"'):
            f, c = reg, STR
        elif tok[0].isdigit():
            f, c = reg, NUM
        elif tok[0].isalpha() or tok[0] == "_" or tok.isspace():
            f, c = reg, INK
        else:
            f, c = reg, PUNCT
        fr.text(x, y, tok, f, c)
        x += fr.width(tok, f)
    return x


def bar_cursor(fr, x, y, size, color=INK):
    fr.box(x, y - 0.80 * size, x + 3, y + 0.26 * size, color)


# --- data ---------------------------------------------------------------------

RACE = json.load(open(os.path.join(HERE, "data", "race.json")))
OUT = open(os.path.join(HERE, "data", "race_output.txt")).read().rstrip("\n").split("\n")


# --- scene 1: change one import -----------------------------------------------

EDIT_WORD = "scimetal"
KS_EDIT = keystrokes(EDIT_WORD, 1.15, seed=3, base=0.075)


def s_edit(fr, t):
    size, x0, ys = 46, 298, (440, 514, 588)
    cw = fr.width("0", F("mono", size))
    if t < 1.15:
        line1 = "import scipy.ndimage as ndimage"
    else:
        line1 = "import " + typed(EDIT_WORD, KS_EDIT, t) + ".ndimage as ndimage"
    if 0.7 <= t < 1.15:  # double-click selects "scipy"
        fr.box(x0 + 7 * cw - 2, ys[0] - 0.80 * size, x0 + 12 * cw + 2, ys[0] + 0.26 * size, mix(ACCENT, 0.22))
    code_line(fr, x0, ys[0], line1, size)
    code_line(fr, x0, ys[2], "smooth = ndimage.gaussian_filter(image, sigma=3)", size)
    if t >= 1.15:
        n = len(typed(EDIT_WORD, KS_EDIT, t))
        idle = KS_EDIT[-1] if n == len(EDIT_WORD) else t
        if cursor_on(t, idle):
            bar_cursor(fr, x0 + (7 + n) * cw, ys[0], size)
    a = ease_out(ramp(t, 2.4, 0.35))
    fr.text(x0, 720, "Change one import.", F("sans", 36, "Medium"), mix(MUTED, a))


# --- scene 2: title -----------------------------------------------------------

def s_title(fr, t):
    fr.text(960, 500, "SciMetal", F("sans", 150, "Bold"), INK, "ms")
    a = ease_out(ramp(t, 0.35, 0.4))
    fr.text(960, 592, "SciPy, accelerated on Apple Silicon.", F("sans", 46), mix(MUTED, a), "ms")


# --- scene 3: the race (real per-call timestamps) -------------------------------

RACE_START = 1.2
RACE_CAPTION = 8.6


def s_race(fr, t):
    x0 = 200
    fr.text(x0, 205, "100 Gaussian blurs, back to back.", F("sans", 60, "Semibold"), INK)
    fr.text(x0, 262, "4096 × 4096 float64 image, σ = 3", F("sans", 32), MUTED)
    e = t - RACE_START
    lab, num = F("mono", 32), F("mono", 32, "Medium")
    bx0, bx1, bh = 560, 1340, 22
    rows = [("scipy.ndimage", RACE["scipy"], GREY, 470), ("scimetal.ndimage", RACE["scimetal"], ACCENT, 590)]
    for name, stamps, color, y in rows:
        n = bisect.bisect_right(stamps, e) if e > 0 else 0
        fr.text(x0, y, name, lab, INK, "lm")
        fr.box(bx0, y - bh / 2, bx1, y + bh / 2, TRACK, r=bh / 2)
        if n:
            fr.box(bx0, y - bh / 2, bx0 + max(bh, (bx1 - bx0) * n / 100), y + bh / 2, color, r=bh / 2)
        fr.text(1380, y, f"{n}/100", num, INK, "lm")
        done = n == len(stamps)
        shown = stamps[-1] if done else max(0.0, e)
        fr.text(1720, y, f"{shown:.2f} s", num, ACCENT if done else MUTED, "rm")

    a = ease_out(ramp(t, RACE_CAPTION, 0.4))
    f, fb = F("sans", 40, "Medium"), F("sans", 40, "Semibold")
    fr.rich(x0, 790, [
        (f"SciPy needs {RACE['scipy'][-1]:.1f} s for all 100. SciMetal took ", f, mix(INK, a)),
        (f"{RACE['scimetal'][-1]:.2f} s", fb, mix(ACCENT, a)),
        (".", f, mix(INK, a)),
    ])
    fr.text(x0, 1010, "Real time. Replayed from per-call timestamps recorded on an Apple M1 Ultra "
            "with scimetal 0.1.0 from PyPI.", F("sans", 22), MUTED)


# --- scene 4: terminal replay ---------------------------------------------------

TERM_CMD = "python race.py"
KS_TERM = keystrokes(TERM_CMD, 0.5, seed=11, base=0.07)
ENTER = KS_TERM[-1] + 0.3
SPEED = 3
# When each output line appeared in the real run, in seconds after Enter: imports, then
# 8 calls per library per case (1 warm-up + 7 timed) at the measured medians, then strict.
REAL_OFFSETS = [0.6, 0.6, 3.86, 10.43, 15.05, 15.52, 15.52]
TERM_DONE = ENTER + REAL_OFFSETS[-1] / SPEED
TERM_HILITE = TERM_DONE + 0.6


def s_term(fr, t):
    fr.box(150, 215, 1770, 865, TERM_BG, r=18)
    f, fb = F("mono", 34), F("mono", 34, "Semibold")
    cw = fr.width("0", f)
    x0, y0, lh = 214, 290, 58
    prompt = "~/demo % "
    fr.text(x0, y0, prompt, f, TERM_DIM)
    cmd = typed(TERM_CMD, KS_TERM, t)
    fr.text(x0 + len(prompt) * cw, y0, cmd, f, TERM_FG)

    n = sum(1 for off in REAL_OFFSETS if t >= ENTER + off / SPEED) if t >= ENTER else 0
    hi = ramp(t, TERM_HILITE, 0.3)
    for i in range(n):
        y = y0 + lh * (i + 1)
        fr.text(x0, y, OUT[i], f, TERM_FG)
        if hi > 0:
            for m in re.finditer(r"\d+x\b|True", OUT[i]):
                cx = x0 + m.start() * cw
                fr.box(cx, y - 34, cx + len(m.group()) * cw, y + 12, TERM_BG)
                fr.text(cx, y, m.group(), fb, mix(TERM_ACCENT, hi, TERM_FG))

    if t < ENTER:
        cx, cy = x0 + (len(prompt) + len(cmd)) * cw, y0
        idle = KS_TERM[-1] if len(cmd) == len(TERM_CMD) else t
        if t < KS_TERM[0]:
            idle = 0.0
    elif n < len(OUT):
        cx, cy, idle = x0, y0 + lh * (n + 1), t
    else:
        cy = y0 + lh * (len(OUT) + 1)
        fr.text(x0, cy, prompt, f, TERM_DIM)
        cx, idle = x0 + len(prompt) * cw, TERM_DONE
    if cursor_on(t, idle):
        fr.box(cx, cy - 34, cx + cw, cy + 12, mix(TERM_FG, 0.85, TERM_BG))

    fr.text(1770, 920, f"Terminal output shown at {SPEED}× speed.", F("sans", 22), MUTED, "rs")


# --- scene 5: strict mode -----------------------------------------------------------

STRICT_WITH = 'with scimetal.precision("strict"):'
STRICT_BODY = "exact = ndimage.gaussian_filter(image, sigma=3)"
KS_STRICT = keystrokes(STRICT_WITH, 0.9, seed=21, base=0.055)
STRICT_TAB = KS_STRICT[-1] + 0.45
STRICT_CAPTION = STRICT_TAB + 0.7


def s_strict(fr, t):
    x0, size = 298, 42
    cw = fr.width("0", F("mono", size))
    y1, y2 = 450, 520
    fr.text(x0, 300, "Need SciPy’s exact answer?", F("sans", 60, "Semibold"), INK)
    if t < 0.6:  # the existing line, before a new line is opened above it
        code_line(fr, x0, y1, STRICT_BODY, size)
        cur, idle = (x0, y1), 0.0
    else:
        head = typed(STRICT_WITH, KS_STRICT, t)
        code_line(fr, x0, y1, head, size)
        if t < STRICT_TAB:
            code_line(fr, x0, y2, STRICT_BODY, size)
            idle = KS_STRICT[-1] if len(head) == len(STRICT_WITH) else (0.6 if t < KS_STRICT[0] else t)
            cur = (x0 + len(head) * cw, y1)
        else:  # tab indents the existing line into the block
            code_line(fr, x0, y2, "    " + STRICT_BODY, size)
            cur, idle = (x0 + 4 * cw, y2), STRICT_TAB
    if cursor_on(t, idle):
        bar_cursor(fr, cur[0], cur[1], size)

    a = ease_out(ramp(t, STRICT_CAPTION, 0.4))
    f, fb = F("sans", 42, "Medium"), F("sans", 42, "Semibold")
    fr.rich(x0, 690, [
        ("Bit-for-bit identical to SciPy 1.18.1. Still ", f, mix(INK, a)),
        ("55× faster", fb, mix(ACCENT, a)),
        (".", f, mix(INK, a)),
    ])
    fr.text(x0, 1010, "gaussian_filter on a 4096 × 4096 float64 image: 7.3 ms on SciMetal’s CPU engine, "
            "404.2 ms in SciPy.", F("sans", 22), mix(MUTED, a))


# --- scene 6: chart (README benchmark table) -------------------------------------

CHART = [
    ("ndimage.gaussian_laplace", "4096 × 4096", 192.8),
    ("signal.convolve2d", "4096 × 4096 image, 5 × 5 kernel", 171.1),
    ("signal.convolve", "2²⁴ samples, 1025 taps", 130.7),
    ("ndimage.gaussian_filter", "4096 × 4096", 105.6),
    ("signal.fftconvolve", "two 4096 × 4096 arrays", 55.3),
    ("fft.fft", "2²⁴ points, complex128", 37.4),
]


def chart_bar_time(i):
    return 0.5 + 0.14 * i


def s_chart(fr, t):
    x0 = 200
    fr.text(x0, 190, "Speedup over SciPy", F("sans", 60, "Semibold"), INK)
    fr.text(x0, 245, "Apple M1 Ultra, float64 unless noted, SciPy 1.18.1 on one thread",
            F("sans", 30), MUTED)
    bx0, bx1, bh = 800, 1600, 30
    vmax = max(v for *_, v in CHART)
    for i, (name, detail, v) in enumerate(CHART):
        y = 370 + i * 108
        fr.text(x0, y, name, F("mono", 30), INK)
        fr.text(x0, y + 36, detail, F("sans", 22), MUTED)
        p = ease_out(ramp(t, chart_bar_time(i), 0.8))
        if p > 0:
            yc = y + 6
            fr.box(bx0, yc - bh / 2, bx0 + max(4, (bx1 - bx0) * v / vmax * p), yc + bh / 2, ACCENT, r=3)
            a = ramp(t, chart_bar_time(i) + 0.7, 0.2)
            fr.text(bx0 + (bx1 - bx0) * v / vmax + 18, yc, f"{round(v)}×",
                    F("sans", 34, "Semibold"), mix(INK, a), "lm")
    fr.text(x0, 1010, "Medians of warm runs from the SciMetal benchmark suite. With 16 SciPy workers: "
            "FFTs 2.5 to 4×, fftconvolve 7×, convolve 33×.", F("sans", 22), MUTED)


# --- scene 7: coverage and tests ----------------------------------------------------

COVER_BEAT = 3.4


def s_cover(fr, t):
    x0 = 200
    big, body = F("sans", 130, "Bold"), F("sans", 46)
    if t < COVER_BEAT:
        fr.text(x0, 500, "32 functions", big, INK)
        fr.text(x0, 592, "in ndimage, fft and signal run natively.", body, INK)
        fr.text(x0, 654, "Everything else falls through to SciPy, untouched.", body, MUTED)
    else:
        fr.text(x0, 500, "14,000+", big, INK)
        fr.text(x0, 592, "of SciPy’s own tests pass in strict mode.", body, INK)
        fr.text(x0, 654, "2,479 ndimage filter tests and 11,690 signal tests.", body, MUTED)


# --- scene 8: install ---------------------------------------------------------------

PIP = "pip install scimetal"
KS_PIP = keystrokes(PIP, 0.5, seed=31, base=0.07)
END_URL = KS_PIP[-1] + 0.5


def s_end(fr, t):
    fr.text(960, 300, "SciMetal", F("sans", 72, "Bold"), INK, "ms")
    fr.box(560, 400, 1360, 500, TERM_BG, r=14)
    f = F("mono", 36)
    cw = fr.width("0", f)
    x = 960 - (len(PIP) + 2) * cw / 2
    y = 463
    fr.text(x, y, "% ", f, TERM_DIM)
    cmd = typed(PIP, KS_PIP, t)
    fr.text(x + 2 * cw, y, cmd, f, TERM_FG)
    idle = KS_PIP[-1] if len(cmd) == len(PIP) else (0.0 if t < KS_PIP[0] else t)
    if cursor_on(t, idle):
        cx = x + (2 + len(cmd)) * cw
        fr.box(cx, y - 36, cx + cw, y + 12, mix(TERM_FG, 0.85, TERM_BG))
    a = ease_out(ramp(t, END_URL, 0.4))
    fr.text(960, 612, "pypi.org/project/scimetal", F("sans", 42, "Medium"), mix(INK, a), "ms")
    fr.text(960, 670, "Apple Silicon  ·  macOS 15+  ·  Python 3.11+", F("sans", 30), mix(MUTED, a), "ms")


# --- timeline -----------------------------------------------------------------------

SCENES = [
    (s_edit, 4.6),
    (s_title, 3.0),
    (s_race, 12.5),
    (s_term, TERM_DONE + 2.8),
    (s_strict, STRICT_TAB + 3.4),
    (s_chart, 8.0),
    (s_cover, 6.8),
    (s_end, KS_PIP[-1] + 4.0),
]

FRAMES = []
for si, (_, dur) in enumerate(SCENES):
    FRAMES += [(si, k / FPS) for k in range(round(dur * FPS))]


def render(i):
    si, lt = FRAMES[i]
    fr = Frame()
    SCENES[si][0](fr, lt)
    return fr.out().tobytes()


def frame_at(t):
    return min(round(t * FPS), len(FRAMES) - 1)


def main():
    start = 0.0
    for fn, dur in SCENES:
        print(f"{start:6.2f}s  {fn.__name__:9s} {dur:5.2f}s")
        start += dur
    print(f"total {start:.2f}s, {len(FRAMES)} frames")

    if "--still" in sys.argv:
        os.makedirs(os.path.join(HERE, "stills"), exist_ok=True)
        for arg in sys.argv[sys.argv.index("--still") + 1:]:
            t = float(arg)
            img = Image.frombytes("RGB", (W, H), render(frame_at(t)))
            path = os.path.join(HERE, "stills", f"still-{t:05.2f}.png")
            img.save(path)
            print(path)
        return

    out = os.path.join(HERE, "scimetal-launch-silent.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-c:v", "libx264", "-preset", "slow", "-crf", "16",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-movflags", "+faststart", out,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(max(1, (os.cpu_count() or 4) - 2)) as pool:
        for k, buf in enumerate(pool.imap(render, range(len(FRAMES)), chunksize=4)):
            proc.stdin.write(buf)
            if k % 300 == 0:
                print(f"frame {k}/{len(FRAMES)}", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")
    print(out)


if __name__ == "__main__":
    main()
