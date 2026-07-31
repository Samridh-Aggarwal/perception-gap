# Validate the SR metric against a synthetic encoder where the right answer is known.
# Plant a large shared bias direction, then check the readout behaves as the geometry says:
# raw shifts get compressed by 1/(rho^2+1), SR itself survives that compression, centering
# lifts a buried signal back off the floor, and an encoder without a bias is left alone.
# Run with: python test_metric.py

import numpy as np

PERTURBATIONS = {"occlude_center": "meaningful", "scramble_patches": "meaningful",
                 "rotate_90": "cosmetic", "flip_vertical": "cosmetic",
                 "matched_noise": "cosmetic"}

def cosine_shifts(base, pert, center=True):
    mu = base.mean(0, keepdims=True) if center else 0.0
    b = base - mu
    bn = np.linalg.norm(b, axis=1)
    return {k: 1.0 - (b * (p - mu)).sum(1) / (bn * np.linalg.norm(p - mu, axis=1) + 1e-12)
            for k, p in pert.items()}

def sr(shifts, idx=None):
    def g(kind):
        return float(np.mean([(s if idx is None else s[idx]).mean()
                              for k, s in shifts.items() if PERTURBATIONS[k] == kind]))
    sig, cos = g("meaningful"), g("cosmetic")
    return sig, cos, sig / (sig + cos)

def bootstrap_ci(shifts, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(next(iter(shifts.values())))
    return np.percentile([sr(shifts, rng.integers(0, N, N))[2] for _ in range(n)], [2.5, 97.5])


rng = np.random.default_rng(0)
N, D = 300, 64
v = rng.normal(size=(N, D))
bias = np.zeros(D); bias[0] = 40.0

def synth(step_meaningful, step_cosmetic, with_bias=True):
    off = bias if with_bias else 0.0
    pert = {k: v + (step_meaningful if kind == "meaningful" else step_cosmetic)
            * rng.normal(size=(N, D)) + off for k, kind in PERTURBATIONS.items()}
    return v + off, pert

rho2 = float(np.linalg.norm((v + bias).mean(0)) ** 2 / v.var(0).sum())
print(f"planted bias: rho^2 = {rho2:.1f}, so raw shifts should read {1/(rho2+1):.4f} of centered\n")

for tag, (sm, sc) in [("small shifts", (0.20, 0.07)), ("large shifts", (0.90, 0.30))]:
    b, p = synth(sm, sc)
    raw, cen = sr(cosine_shifts(b, p, center=False)), sr(cosine_shifts(b, p))
    print(f"{tag:<13} signal {raw[0]:.5f} -> {cen[0]:.5f}   compression {raw[0]/cen[0]:.4f}"
          f"   SR {raw[2]:.4f} -> {cen[2]:.4f}")

b, p = synth(0.20, 0.07)
raw, cen = sr(cosine_shifts(b, p, center=False)), sr(cosine_shifts(b, p))
assert abs((raw[0] / cen[0]) / (1 / (rho2 + 1)) - 1) < 0.10, "compression should match 1/(rho^2+1)"
assert abs(raw[2] - cen[2]) < 0.02, "SR should survive centering, the factor cancels in the ratio"
assert cen[0] / raw[0] > 10, "centering should lift a buried signal off the floor"

b0, p0 = synth(0.20, 0.07, with_bias=False)
raw0, cen0 = sr(cosine_shifts(b0, p0, center=False)), sr(cosine_shifts(b0, p0))
print(f"\nno planted bias: signal {raw0[0]:.5f} -> {cen0[0]:.5f}, SR {raw0[2]:.4f} -> {cen0[2]:.4f}")
assert abs(raw0[0] - cen0[0]) < 0.02 and abs(raw0[2] - cen0[2]) < 0.02, \
    "centering should be a no-op on an encoder with no shared bias"

sh = cosine_shifts(b, p)
lo, hi = bootstrap_ci(sh)
assert lo < sr(sh)[2] < hi, "CI should bracket the point estimate"
assert bootstrap_ci(sh)[0] == lo, "CI should be identical across calls"
print(f"bootstrap [{lo:.4f} {hi:.4f}] brackets {sr(sh)[2]:.4f} and repeats exactly")
print("\nall checks pass")
