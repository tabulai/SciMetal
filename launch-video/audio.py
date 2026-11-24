#!/usr/bin/env python3
"""Synthesize a quiet soundtrack for the SciMetal launch video and mux it in.

Every sound is generated here, with no samples or downloaded audio:
  music  a soft pad and a Rhodes-like FM electric piano in F major; chords change on
         scene cuts and on the moments the picture lands (captions, highlights)
  foley  a down- and up-stroke for every typed key, a double-click, a quiet tick for each
         completed SciPy call in the race, one soft note when SciMetal finishes

Cue times come from render.py, so picture and sound stay in sync if the timing changes.

Usage:
  python3 render.py && python3 audio.py      # writes soundtrack.wav and scimetal-launch.mp4
"""
import os
import re
import subprocess

import numpy as np
from scipy import signal
from scipy.io import wavfile

import render as R

SR = 48000
TARGET_LUFS = -20.0   # integrated loudness of the final mix
FADE_OUT = 2.2        # seconds of music fade at the very end

# Relative bus peaks before loudness normalisation. Lower = quieter in the mix.
PAD_PEAK = 0.14
EPIANO_PEAK = 0.50
KEYS_PEAK = 0.22
MOUSE_PEAK = 0.16
TICK_PEAK = 0.055

SCENE = {}
_t = 0.0
for _fn, _dur in R.SCENES:
    SCENE[_fn.__name__] = _t
    _t += _dur
TOTAL = _t
N = int(round(TOTAL * SR))
TAIL = int(3.0 * SR)  # room for reverb tails before the final trim


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def place(bus, x, t, pan=0.0, gain=1.0):
    """Add mono or stereo buffer x into a stereo bus at time t (seconds)."""
    i = int(round(t * SR))
    if i >= bus.shape[1]:
        return
    if x.ndim == 1:
        left, right = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        x = np.vstack([x * left, x * right]) * np.sqrt(2)
    n = min(x.shape[1], bus.shape[1] - i)
    bus[:, i:i + n] += gain * x[:, :n]


def peak_to(x, peak):
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


# --- harmony ------------------------------------------------------------------------

VOICING = {  # bass note, upper structure (MIDI)
    "F": (41, [57, 60, 64, 67]),      # Fmaj9
    "Dm": (38, [53, 57, 60, 64]),     # Dm9
    "Bb": (34, [57, 60, 62, 65]),     # Bbmaj9
    "Gm": (43, [58, 62, 65, 69]),     # Gm9
    "Csus": (36, [58, 62, 65, 67]),   # C9sus4
}
MELODY = {"F": [72, 76], "Dm": [69, 72], "Bb": [74, 72], "Gm": [70, 74], "Csus": [74, 72]}

race = SCENE["s_race"]
term = SCENE["s_term"]
strict = SCENE["s_strict"]
chart = SCENE["s_chart"]
cover = SCENE["s_cover"]
end = SCENE["s_end"]

# (time, chord, offsets of soft melody notes after the chord). The first entry has no
# piano: the intro is just the pad fading in under the typing.
CHANGES = [
    (0.0, "F", None),
    (SCENE["s_title"], "F", [1.6]),
    (race, "Dm", [4.9]),
    (race + R.RACE_CAPTION, "Bb", [1.8]),
    (term, "F", [1.6, 3.6]),
    (term + R.TERM_HILITE, "Csus", []),
    (strict, "Dm", [2.0]),
    (strict + R.STRICT_CAPTION, "Bb", [1.4]),
    (chart, "Gm", []),
    (chart + 4.0, "Csus", [1.6]),
    (cover, "F", [1.6]),
    (cover + R.COVER_BEAT, "Dm", [1.6]),
    (end, "Bb", [1.2]),
    (end + R.END_URL, "F", [1.6]),
]


# --- instruments --------------------------------------------------------------------

def pad(rng):
    """Soft additive pad, one detuned voice per channel, crossfading between chords."""
    bus = np.zeros((2, N + TAIL))
    t_all = np.arange(N + TAIL) / SR
    segs = []
    for t, chord, _ in CHANGES:  # merge repeated chords so the pad does not re-attack
        if segs and segs[-1][1] == chord:
            continue
        segs.append([t, chord])
    for k, (start, chord) in enumerate(segs):
        stop = segs[k + 1][0] if k + 1 < len(segs) else TOTAL + 2.0
        attack = 2.5 if k == 0 else 1.2
        a0 = max(0.0, start - (0.0 if k == 0 else 0.15))
        env = np.clip((t_all - a0) / attack, 0, 1)
        env = np.minimum(env, np.clip(1 - (t_all - stop) / 1.8, 0, 1))
        env = 0.5 - 0.5 * np.cos(np.pi * env)
        live = env > 0
        t = t_all[live]
        bass, upper = VOICING[chord]
        for ch, cents in ((0, -4.0), (1, 4.0)):
            y = np.zeros(t.size)
            for m in upper:
                f = mtof(m) * 2 ** (cents / 1200)
                ph = rng.uniform(0, 2 * np.pi)
                drift = 1 + 0.08 * np.sin(2 * np.pi * rng.uniform(0.06, 0.14) * t + rng.uniform(0, 2 * np.pi))
                for h in range(1, 7):
                    y += drift * np.sin(2 * np.pi * f * h * t + ph * h) / h ** 1.7
            fb = mtof(bass)
            y += 0.45 * (np.sin(2 * np.pi * fb * t) + 0.25 * np.sin(4 * np.pi * fb * t))
            bus[ch, live] += y * env[live]
    sos = signal.butter(4, [70, 1400], btype="bandpass", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, bus, axis=1)


def epiano_note(midi, vel, dur=5.0):
    """Two-operator FM (1:1) with a decaying index: the classic soft electric piano."""
    f = mtof(midi)
    t = np.arange(int(dur * SR)) / SR
    index = (0.35 + 2.0 * vel) * np.exp(-t / 0.45) + 0.12
    tau = 1.9 * (261.6 / f) ** 0.35
    env = (1 - np.exp(-t / 0.0025)) * np.exp(-t / tau)
    y = np.sin(2 * np.pi * f * t + index * np.sin(2 * np.pi * f * t))
    if f * 14 < 15000:  # brief tine "ping" on the attack
        y += 0.05 * vel * np.sin(2 * np.pi * f * 14 * t) * np.exp(-t / 0.025)
    return vel * env * y


def epiano(rng):
    bus = np.zeros((2, N + TAIL))

    def note(t, m, vel):
        place(bus, epiano_note(m, vel), t, pan=np.clip((m - 64) / 40, -0.4, 0.4))

    for t, chord, melody in CHANGES:
        if melody is None:
            continue
        bass, upper = VOICING[chord]
        for j, m in enumerate([bass + 12] + upper):  # gentle upward roll
            note(t + j * rng.uniform(0.018, 0.03), m, rng.uniform(0.2, 0.27) * (0.85 if j == 0 else 1))
        for k, off in enumerate(melody):
            note(t + off + rng.uniform(-0.02, 0.02), MELODY[chord][k % 2], rng.uniform(0.19, 0.24))

    # SciMetal finishing its 100 calls: one soft dyad (chord tones of Dm9).
    done = race + R.RACE_START + R.RACE["scimetal"][-1]
    note(done, 81, 0.24)
    note(done + 0.07, 88, 0.2)

    # Chart bars: a quiet descending Gm9 figure, one note as each bar starts to grow.
    for i, m in enumerate([81, 77, 74, 70, 69, 67]):
        note(chart + R.chart_bar_time(i), m, 0.15)

    # 1:1 FM puts a sideband at 0 Hz; the high-pass removes that thump on each note.
    sos = signal.butter(2, [60, 5000], btype="bandpass", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, bus, axis=1)


# --- foley --------------------------------------------------------------------------

KEY_NOISE = signal.butter(2, [1500, 6500], btype="bandpass", fs=SR, output="sos")
MOUSE_NOISE = signal.butter(2, [2500, 9000], btype="bandpass", fs=SR, output="sos")


def stroke(rng, n, noise_tau, body_f, body_tau, noise_sos, body_amp=0.5):
    t = np.arange(n) / SR
    noise = signal.sosfilt(noise_sos, rng.standard_normal(n)) * np.exp(-t / noise_tau)
    body = np.sin(2 * np.pi * body_f * t) * np.exp(-t / body_tau)
    return noise / (np.max(np.abs(noise)) + 1e-9) * 0.7 + body_amp * body


def keystroke(rng, kind="key"):
    """Down-stroke plus a quieter up-stroke, the way a real key sounds."""
    n = int(0.06 * SR)
    lo, hi = {"key": (170, 260), "space": (105, 140), "enter": (120, 160)}[kind]
    down = stroke(rng, n, rng.uniform(0.0025, 0.004), rng.uniform(lo, hi), 0.011, KEY_NOISE,
                  body_amp=0.7 if kind != "key" else 0.5)
    up = stroke(rng, n, 0.002, rng.uniform(lo, hi) * 1.4, 0.006, KEY_NOISE, body_amp=0.3)
    gap = int(rng.uniform(0.07, 0.11) * SR) if kind == "key" else int(rng.uniform(0.11, 0.15) * SR)
    out = np.zeros(gap + n)
    out[:n] += down * (1.25 if kind != "key" else 1.0)
    out[gap:] += up * rng.uniform(0.3, 0.42)
    return out * rng.uniform(0.7, 1.0)


def mouse_click(rng):
    n = int(0.03 * SR)
    down = stroke(rng, n, 0.0012, rng.uniform(1600, 2000), 0.003, MOUSE_NOISE, body_amp=0.4)
    up = stroke(rng, n, 0.001, 2300, 0.002, MOUSE_NOISE, body_amp=0.25)
    gap = int(0.055 * SR)
    out = np.zeros(gap + n)
    out[:n] += down
    out[gap:] += 0.45 * up
    return out


def tick(rng):
    """A small, dry woodblock-like tick."""
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    y = np.sin(2 * np.pi * 1150 * t) * np.exp(-t / 0.016) + 0.5 * np.sin(2 * np.pi * 1720 * t) * np.exp(-t / 0.009)
    y += 0.15 * signal.sosfilt(MOUSE_NOISE, rng.standard_normal(n)) * np.exp(-t / 0.0015)
    return y * rng.uniform(0.85, 1.0)


def typing(bus, rng, text, stamps, offset):
    for ch, t in zip(text, stamps):
        place(bus, keystroke(rng, "space" if ch == " " else "key"), offset + t, pan=rng.uniform(-0.2, 0.2))


def foley(rng):
    keys = np.zeros((2, N + TAIL))
    mouse = np.zeros((2, N + TAIL))
    ticks = np.zeros((2, N + TAIL))

    # Scene 1: double-click selects "scipy" (the highlight appears on the second click).
    for t in (0.58, 0.70):
        place(mouse, mouse_click(rng), t, pan=0.15)
    typing(keys, rng, R.EDIT_WORD, R.KS_EDIT, 0.0)

    # Race: one tick per completed SciPy call, until the scene cuts away.
    start = race + R.RACE_START
    for s in R.RACE["scipy"]:
        if start + s < term - 0.05:
            place(ticks, tick(rng), start + s, pan=-0.1)

    typing(keys, rng, R.TERM_CMD, R.KS_TERM, term)
    place(keys, keystroke(rng, "enter"), term + R.ENTER - 0.05, pan=0.1)

    place(keys, keystroke(rng, "enter"), strict + 0.6 - 0.04)  # open a line above
    typing(keys, rng, R.STRICT_WITH, R.KS_STRICT, strict)
    place(keys, keystroke(rng, "enter"), strict + R.STRICT_TAB - 0.04, pan=-0.15)  # tab

    typing(keys, rng, R.PIP, R.KS_PIP, end)

    return peak_to(keys, KEYS_PEAK) + peak_to(mouse, MOUSE_PEAK) + peak_to(ticks, TICK_PEAK)


# --- space --------------------------------------------------------------------------

def impulse_response(rng, rt60_low, rt60_high, length, predelay):
    n = int(length * SR)
    t = np.arange(n) / SR
    ir = np.zeros((2, n + int(predelay * SR)))
    lp = signal.butter(2, 2500, btype="lowpass", fs=SR, output="sos")
    hp = signal.butter(2, 2500, btype="highpass", fs=SR, output="sos")
    for ch in range(2):
        noise = rng.standard_normal(n)
        y = signal.sosfilt(lp, noise) * np.exp(-6.9 * t / rt60_low)
        y += signal.sosfilt(hp, noise) * np.exp(-6.9 * t / rt60_high)
        ir[ch, int(predelay * SR):] = y
    return ir / np.sqrt(np.sum(ir ** 2) / 2)


def reverb(x, ir, wet):
    out = np.vstack([signal.fftconvolve(x[ch], ir[ch])[: x.shape[1]] for ch in range(2)])
    return x + wet * out


# --- mix and mux --------------------------------------------------------------------

def loudness(path):
    res = subprocess.run(["ffmpeg", "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"],
                         capture_output=True, text=True)
    summary = res.stderr[res.stderr.rfind("Summary:"):]
    lufs = float(re.search(r"I:\s+(-?[\d.]+) LUFS", summary).group(1))
    peak = float(re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", summary).group(1))
    return lufs, peak


def main():
    rng = np.random.default_rng(7)
    music = peak_to(pad(rng), PAD_PEAK) + peak_to(epiano(rng), EPIANO_PEAK)
    music = reverb(music, impulse_response(rng, 2.6, 1.0, 3.0, 0.025), wet=0.35)
    fx = reverb(foley(rng), impulse_response(rng, 0.35, 0.2, 0.5, 0.004), wet=0.12)

    t = np.arange(N + TAIL) / SR
    fade = np.clip((TOTAL - t) / FADE_OUT, 0, 1)
    music *= 0.5 - 0.5 * np.cos(np.pi * fade)
    mix = (music + fx)[:, :N]
    mix = signal.sosfiltfilt(signal.butter(2, 35, btype="highpass", fs=SR, output="sos"), mix, axis=1)
    mix = peak_to(mix, 0.5)

    wav = os.path.join(R.HERE, "soundtrack.wav")
    wavfile.write(wav, SR, mix.T.astype(np.float32))
    lufs, peak = loudness(wav)
    gain = TARGET_LUFS - lufs
    print(f"soundtrack: {TOTAL:.2f} s, {lufs:.1f} LUFS, true peak {peak:.1f} dBFS; applying {gain:+.1f} dB")
    if peak + gain > -1.0:
        gain = -1.0 - peak
        print(f"  capped gain at {gain:+.1f} dB to keep the true peak under -1 dBFS")

    video = os.path.join(R.HERE, "scimetal-launch-silent.mp4")
    out = os.path.join(R.HERE, "scimetal-launch.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", wav,
        "-map", "0:v", "-map", "1:a", "-c:v", "copy",
        "-af", f"volume={gain:.2f}dB", "-c:a", "aac", "-b:a", "192k", "-ar", str(SR),
        "-shortest", "-movflags", "+faststart", out,
    ], check=True)
    print(out)


if __name__ == "__main__":
    main()
