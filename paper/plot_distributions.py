"""
Illustrate the different types of distributions used for significance test comparisons.
"""

# EXT
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import laplace, norm, rayleigh


if __name__ == "__main__":
    x = np.linspace(-3, 6, 9000)

    plt.figure(figsize=(8, 6))
    plt.rcParams.update(
        {"font.size": 18, "text.usetex": True, "legend.loc": "upper right"}
    )

    ax = plt.gca()
    # ax.set_ylim(0, 1)
    ax.yaxis.grid()
    ax.xaxis.grid()

    # Plot normal
    plt.plot(
        x,
        norm.pdf(x, loc=0.5, scale=1.5),
        label="Normal",
        alpha=0.8,
        color="tab:blue",
        linewidth=1.4,
    )

    # Plot normal mixture
    plt.plot(
        x,
        norm.pdf(x, loc=-0.5, scale=0.25) * 0.3 + norm.pdf(x, loc=2.5, scale=1.0) * 0.7,
        label="Normal Mixture",
        alpha=0.8,
        color="tab:orange",
        linewidth=1.4,
    )

    # Plot Laplace
    plt.plot(
        x,
        laplace.pdf(x, loc=0.5, scale=1.5),
        label="Laplace",
        alpha=0.8,
        color="tab:green",
        linewidth=1.4,
    )

    # Plot Rayleigh
    plt.plot(
        x,
        rayleigh.pdf(x, scale=2),
        label="Rayleigh",
        alpha=0.8,
        color="tab:red",
        linewidth=1.4,
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig("img/distributions.png")
