# interleaved A/B of plans (robust to drifting background load)
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, scipy.fft
import e5lib
from e5lib import Plan
e5lib.init()
rng = np.random.default_rng(0)
lg = int(sys.argv[1]); cands = json.loads(sys.argv[2]); rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 12
rows = int(sys.argv[4]) if len(sys.argv) > 4 else 1
n = 1 << lg
x = (rng.standard_normal((rows, n)) + 1j*rng.standard_normal((rows, n))).astype(np.complex64)
plans = []
for c in cands:
    f, W = (c[0], c[1]) if c else (None, 0)
    p = Plan(n, rows=rows, factors=f, coalesce=W) if f else Plan(n, rows=rows)
    p.x[:] = x
    for _ in range(3): p.exec()
    plans.append((str(f or "mlx-plan") + " W%d" % W, p, [], []))
for r in range(rounds):
    for name, p, ts, gs in plans:
        for _ in range(3):
            ts.append(p.exec()); gs.append(p.gpu_seconds())
for name, p, ts, gs in plans:
    print("2^%d x%d %-34s wall med %.3f  p25 %.3f  min %.3f | gpu med %.3f min %.3f ms" % (lg, rows, name, 1e3*np.median(ts), 1e3*np.percentile(ts, 25), 1e3*min(ts), 1e3*np.median(gs), 1e3*min(gs)))
