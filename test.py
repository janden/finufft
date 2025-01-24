from cufinufft import Plan
import cupy as cp

import numpy as np

for i in range(30):
    M, N = 4096, 4096

    # Need to alternate in precision to trigger this.
    if i % 2 == 0:
        real_dtype = np.float32
    else:
        real_dtype = np.float64

    complex_dtype = np.complex64 if real_dtype == np.float32 else np.complex128

    plan = Plan(3, 2, dtype=complex_dtype)

    cp.random.seed(0)

    x = cp.random.uniform(-np.pi, np.pi, M, dtype=real_dtype)
    y = cp.random.uniform(-np.pi, np.pi, M, dtype=real_dtype)

    c = (cp.random.standard_normal(size=M, dtype=real_dtype)
         + 1J * cp.random.standard_normal(size=M, dtype=real_dtype))

    s = cp.random.uniform(-np.pi, np.pi, M, dtype=real_dtype)
    t = cp.random.uniform(-np.pi, np.pi, M, dtype=real_dtype)

    plan.setpts(x, y, None, s, t, None)

    f = plan.execute(c)

    f_true = cp.exp(1j * (s[:, None] * x[None] + t[:, None] * y[None])) @ c

    rel_err = (cp.abs(f - f_true).get().max() / cp.abs(f_true).get().max())

    ind = cp.abs(f- f_true).get().argmax()

    if rel_err > 1e6:
        print(f"extreme rel_err = {rel_err}")
        print(f"ind = {ind}")
