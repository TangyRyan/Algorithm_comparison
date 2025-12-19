import os
import numpy as np

Ns = [100, 2000, 3000, 4000, 5000]
SEEDS = [0, 1, 2, 3, 4]  # 你想要几次重复就填几个

def main():
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        out_root = os.path.join("data_seeds", f"seed_{seed}")
        os.makedirs(out_root, exist_ok=True)

        for N in Ns:
            d = os.path.join(out_root, str(N))
            os.makedirs(d, exist_ok=True)

            A = rng.random((N, 2), dtype=np.float32)
            B = rng.random((N, 2), dtype=np.float32)

            np.save(os.path.join(d, "A.npy"), A)
            np.save(os.path.join(d, "B.npy"), B)

            print(f"saved seed={seed} N={N} -> {d}")

if __name__ == "__main__":
    main()
