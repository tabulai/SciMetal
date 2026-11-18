import sys, os, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, scipy.fft
import e5lib
from e5lib import Plan
e5lib.init()
rng = np.random.default_rng(0)
lg = int(sys.argv[1]) if len(sys.argv) > 1 else 24
ks = [int(a) for a in sys.argv[2].split(",")] if len(sys.argv) > 2 else [3]
Ws = [int(a) for a in sys.argv[3].split(",")] if len(sys.argv) > 3 else [4, 16, 64, 256]
n = 1 << lg
x = (rng.standard_normal(n) + 1j*rng.standard_normal(n)).astype(np.complex64)
ref = scipy.fft.fft(x.astype(np.complex128))
nref = np.linalg.norm(ref)
def bench(p, k=7):
    p.x[0] = x
    for _ in range(2): p.exec()
    err = np.linalg.norm(p.y[0] - ref)/nref
    ts = sorted(p.exec() for _ in range(k))
    p.set_split_cb(1)
    pp = []
    for _ in range(5):
        p.exec(); pp.append(p.pass_gpu_seconds())
    p.set_split_cb(0)
    pp = np.median(np.array(pp), axis=0)
    return 1e3*ts[k//2], 1e3*pp, err
res = []
LO=int(os.environ.get('LO','2')); HI=int(os.environ.get('HI','12'))
def comps(total, k, lo=LO, hi=HI):
    if k == 1:
        if lo <= total <= hi: yield (total,)
        return
    for a in range(lo, hi+1):
        for rest in comps(total - a, k-1, lo, hi):
            yield (a,) + rest
for k in ks:
    for c in comps(lg, k):
        f = [1 << a for a in c]
        for W in Ws:
            try:
                p = Plan(n, factors=f, coalesce=W)
            except RuntimeError as e:
                print(f, W, "ERR", str(e)[:100]); continue
            d = p.describe()
            cws = tuple(q["group"][1] for q in d)
            t, pp, err = bench(p)
            res.append((t, tuple(f), W, cws, pp, err))
            p.close()
# dedupe identical (f, cws)
seen = {}
for r in res:
    key = (r[1], r[3])
    if key not in seen or r[0] < seen[key][0]: seen[key] = r
res = sorted(seen.values(), key=lambda r: r[0])
for t, f, W, cws, pp, err in res[:int(os.environ.get("TOP", "25"))]:
    print("%-28s cw %-18s %.2f ms  gpu/pass %s sum %.2f  err %.2e" % (f, cws, t, np.round(pp, 2), pp.sum(), err))
bad = [r for r in res if r[5] > 3e-6]
print("configs", len(res), "bad", len(bad), [ (r[1], r[3], r[5]) for r in bad[:5]])
