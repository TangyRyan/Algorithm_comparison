import os
import numpy as np

Ns = [1000, 5000, 10000, 20000, 30000, 50000, 80000, 100000]

def main(out_root="data", seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(out_root, exist_ok=True)

    for N in Ns:
        d = os.path.join(out_root, str(N))
        os.makedirs(d, exist_ok=True)

        A = rng.random((N, 2), dtype=np.float32)  # 等价于 np.random.rand(N,2) 但可控 seed
        B = rng.random((N, 2), dtype=np.float32)

        np.save(os.path.join(d, "A.npy"), A)
        np.save(os.path.join(d, "B.npy"), B)

        print(f"saved N={N} -> {d}")

if __name__ == "__main__":
    main()
