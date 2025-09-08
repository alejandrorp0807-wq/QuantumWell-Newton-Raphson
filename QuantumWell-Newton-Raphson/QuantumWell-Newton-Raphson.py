"""
Quantum Well in 1D - Newton-Raphson Method
------------------------------------------

This script solves the 1D finite quantum well problem using the
time-independent Schrödinger equation and the Newton-Raphson root-finding method.

It complements the Finite Difference and Numerov implementations by
providing a faster solver with SciPy optimization tools.

Author: Alejandro Rodríguez Peláez
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
import time


# ==============================
# Physical constants
# ==============================
V0 = 244        # Potential depth (eV)
a = 1e-10       # Well width (m)
c = 3e8         # Speed of light (m/s)
me = 0.511e6 / c**2  # Electron mass (MeV/c^2)
hbar = 6.582e-16     # Reduced Planck constant (eV*s)

# Dimensionless parameter
k = (2 * me * a**2 * V0) / hbar**2
print("k =", k)

# ==============================
# Grid setup
# ==============================
umin, umax = -2 * a, 2 * a
npts = 1000
u = np.linspace(umin, umax, npts) / a
du = u[1] - u[0]


# ==============================
# Potential function (rectangular well)
# ==============================
def C(u_val, alpha):
    """Return the coefficient C(u, alpha) for the rectangular well."""
    return -k * alpha if -0.5 < u_val < 0.5 else k * (1 - alpha)


# ==============================
# Finite difference solver for PSI
# ==============================
def psi_boundary(alpha):
    """Return last values of even and odd solutions for given alpha."""
    psi_even, psi_odd, dpsi_even, dpsi_odd = np.zeros((4, npts // 2))
    psi_even[0], dpsi_even[0] = 1, 0
    psi_odd[0], dpsi_odd[0] = 0, 1

    for i in range(1, npts // 2):
        psi_even[i] = psi_even[i - 1] + dpsi_even[i - 1] * du
        dpsi_even[i] = dpsi_even[i - 1] + C(u[i + npts // 2], alpha) * psi_even[i] * du

        psi_odd[i] = psi_odd[i - 1] + dpsi_odd[i - 1] * du
        dpsi_odd[i] = dpsi_odd[i - 1] + C(u[i + npts // 2], alpha) * psi_odd[i] * du

    return psi_even[-1], psi_odd[-1]


def wavefunction(alpha, parity="even"):
    """Return normalized wavefunction (even or odd) for a given alpha."""
    psi, dpsi = np.zeros(npts // 2), np.zeros(npts // 2)

    if parity == "even":
        psi[0], dpsi[0] = 1, 0
    else:
        psi[0], dpsi[0] = 0, 1

    for i in range(1, npts // 2):
        psi[i] = psi[i - 1] + dpsi[i - 1] * du
        dpsi[i] = dpsi[i - 1] + C(u[i + npts // 2], alpha) * psi[i] * du

    if parity == "even":
        full = np.concatenate((psi[::-1], psi))
    else:
        full = np.concatenate((-psi[::-1], psi))

    return full / np.linalg.norm(full)


# ==============================
# Newton-Raphson Method
# ==============================
def func_even(alpha):
    return psi_boundary(alpha)[0]


def func_odd(alpha):
    return psi_boundary(alpha)[1]


def derivative(f, x, h=1e-11):
    """Numerical derivative for Newton method."""
    return (f(x + h) - f(x)) / h


def find_alpha_newton(f, df, guess):
    """Find root using Newton-Raphson."""
    return newton(f, guess, fprime=df)


# ==============================
# Main execution
# ==============================
def main():
    start_time = time.time()

    # Theoretical values
    alpha_theory = [0.09797, 0.3825, 0.8075]
    E_theory = [a * V0 for a in alpha_theory]

    # Solve with Newton-Raphson
    alpha_even1 = find_alpha_newton(func_even, lambda x: derivative(func_even, x), 0.1)
    alpha_even2 = find_alpha_newton(func_even, lambda x: derivative(func_even, x), 0.6)
    alpha_odd1 = find_alpha_newton(func_odd, lambda x: derivative(func_odd, x), 0.3)

    alpha_solutions = [alpha_even1, alpha_odd1, alpha_even2]
    E_newton = [val * V0 for val in alpha_solutions]

    # Errors
    errors = [abs(alpha_theory[i] - alpha_solutions[i]) for i in range(3)]

    elapsed = time.time() - start_time

    # Print results
    print("\n=== Newton-Raphson Results ===")
    print("Theoretical energies:", E_theory)
    print("Newton-Raphson energies:", E_newton)
    print("Alpha solutions:", alpha_solutions)
    print("Errors:", errors)
    print("Execution time:", elapsed, "seconds")

    # ======================
    # Plot wavefunctions
    # ======================
    plt.figure(figsize=(12, 6))

    psi_even = wavefunction(alpha_even1, parity="even")
    plt.subplot(2, 2, 1)
    plt.plot(u, psi_even, label=f"Even α={alpha_even1:.4f}")
    plt.grid()
    plt.legend()

    psi_odd = wavefunction(alpha_odd1, parity="odd")
    plt.subplot(2, 2, 2)
    plt.plot(u, psi_odd, label=f"Odd α={alpha_odd1:.4f}", color="red")
    plt.grid()
    plt.legend()

    psi_even2 = wavefunction(alpha_even2, parity="even")
    plt.subplot(2, 2, 3)
    plt.plot(u, psi_even2, label=f"Even α={alpha_even2:.4f}")
    plt.grid()
    plt.legend()

    plt.suptitle("Newton-Raphson Wavefunctions (Even & Odd States)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
