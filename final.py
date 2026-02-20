
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import hilbert, solve
from numpy.linalg import cond, norm
from datetime import datetime
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 11


class HilbertExperiment:

    def __init__(self, max_size=15):
        self.max_size = max_size
        self.results = []
        self.comp_results = []
        self.df = None
        self.comp_df = None

    # --------------------------------------------------
    # Build system Hx = b with known true solution
    # --------------------------------------------------
    def generate_system(self, n):
        H = hilbert(n)
        x_true = np.ones(n)
        b = H @ x_true
        return H, x_true, b

    # --------------------------------------------------
    # Solve and compute metrics
    # --------------------------------------------------
    def analyze_size(self, n):
        try:
            H, x_true, b = self.generate_system(n)

            cond_number = cond(H)
            x_comp = solve(H, b)

            rel_error = norm(x_comp - x_true, np.inf) / norm(x_true, np.inf)
            residual = norm(H @ x_comp - b, np.inf)

            return {
                "size": n,
                "cond_number": cond_number,
                "log_cond": np.log10(cond_number),
                "rel_error": rel_error,
                "log_error": np.log10(rel_error),
                "residual": residual,
                "digits_lost": np.log10(cond_number),
                "success": True
            }

        except Exception:
            return {
                "size": n,
                "cond_number": np.inf,
                "log_cond": np.inf,
                "rel_error": np.inf,
                "log_error": np.inf,
                "residual": np.inf,
                "digits_lost": np.inf,
                "success": False
            }

    # --------------------------------------------------
    # Run main experiment
    # --------------------------------------------------
    def run(self):
        print("="*70)
        print("Hilbert Matrix Numerical Instability Study")
        print("="*70)
        print(f"Date: {datetime.now()}\n")

        for n in range(2, self.max_size + 1):
            result = self.analyze_size(n)
            self.results.append(result)

            if result["success"]:
                print(
                    f"n={n:2d} | "
                    f"cond={result['cond_number']:.3e} | "
                    f"rel_error={result['rel_error']:.3e}"
                )

        self.df = pd.DataFrame(self.results)
        return self.df

    # --------------------------------------------------
    # Compare with random matrices
    # --------------------------------------------------
    def compare_with_random(self, max_n=10):
        print("\nComparing with random matrices...\n")

        for n in range(2, min(max_n, self.max_size) + 1):

            H_h = hilbert(n)
            H_r = np.random.rand(n, n)
            x_true = np.ones(n)

            cond_h = cond(H_h)
            cond_r = cond(H_r)

            x_h = solve(H_h, H_h @ x_true)
            x_r = solve(H_r, H_r @ x_true)

            err_h = norm(x_h - x_true, np.inf)
            err_r = norm(x_r - x_true, np.inf)

            self.comp_results.append({
                "size": n,
                "hilbert_cond": cond_h,
                "random_cond": cond_r,
                "hilbert_error": err_h,
                "random_error": err_r
            })

            print(
                f"n={n:2d} | "
                f"Hilbert cond={cond_h:.2e} | "
                f"Random cond={cond_r:.2e}"
            )

        self.comp_df = pd.DataFrame(self.comp_results)
        return self.comp_df

    # --------------------------------------------------
    # FIGURE 1: Condition + Error growth
    # --------------------------------------------------
    def plot_main(self):
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        ax[0].semilogy(self.df["size"], self.df["cond_number"], "o-")
        ax[0].set_title("Condition Number Growth")
        ax[0].set_xlabel("Matrix Size")
        ax[0].set_ylabel("Condition Number (log scale)")

        ax[1].semilogy(self.df["size"], self.df["rel_error"], "o-")
        ax[1].set_title("Relative Error Growth")
        ax[1].set_xlabel("Matrix Size")
        ax[1].set_ylabel("Relative Error (log scale)")

        plt.tight_layout()
        plt.savefig("figure_main.png", dpi=300)
        plt.show()

    # --------------------------------------------------
    # FIGURE 2: Error vs Condition (theoretical relation)
    # --------------------------------------------------
    def plot_relation(self):
        plt.figure(figsize=(7, 6))

        plt.scatter(self.df["log_cond"], self.df["log_error"], c=self.df["size"])
        x = np.linspace(0, 20, 100)
        plt.plot(x, x, "--", label="y = x (theoretical bound)")

        plt.xlabel("log10(Condition Number)")
        plt.ylabel("log10(Relative Error)")
        plt.title("Error vs Condition Number")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("figure_relation.png", dpi=300)
        plt.show()

    # --------------------------------------------------
    # FIGURE 3: Comparison with random matrices
    # --------------------------------------------------
    def plot_comparison(self):
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        ax[0].semilogy(self.comp_df["size"], self.comp_df["hilbert_cond"], "o-")
        ax[0].semilogy(self.comp_df["size"], self.comp_df["random_cond"], "o-")
        ax[0].set_title("Condition Number: Hilbert vs Random")
        ax[0].set_xlabel("Matrix Size")

        ax[1].semilogy(self.comp_df["size"], self.comp_df["hilbert_error"], "o-")
        ax[1].semilogy(self.comp_df["size"], self.comp_df["random_error"], "o-")
        ax[1].set_title("Error: Hilbert vs Random")
        ax[1].set_xlabel("Matrix Size")

        plt.tight_layout()
        plt.savefig("figure_comparison.png", dpi=300)
        plt.show()

    # --------------------------------------------------
    # Generate summary report
    # --------------------------------------------------
    def report(self):
        print("\nExperiment Summary")
        print("-"*40)

        max_cond = self.df["cond_number"].max()
        max_error = self.df["rel_error"].max()

        print(f"Largest condition number: {max_cond:.2e}")
        print(f"Worst relative error: {max_error:.2e}")

        critical = self.df[self.df["rel_error"] > 1.0]
        if len(critical) > 0:
            n_fail = int(critical.iloc[0]["size"])
            print(f"Numerical breakdown begins at n = {n_fail}")


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
def main():

    experiment = HilbertExperiment(max_size=15)

    experiment.run()
    experiment.compare_with_random()

    experiment.plot_main()
    experiment.plot_relation()
    experiment.plot_comparison()

    experiment.report()

    return experiment


if __name__ == "__main__":
    main()
