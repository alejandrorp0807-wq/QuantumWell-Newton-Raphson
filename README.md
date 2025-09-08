# QuantumWell-Newton-Raphson

**Newton–Raphson solver for the 1D finite quantum well** — a compact implementation that finds bound-state eigenvalues of the time-independent Schrödinger equation using SciPy's Newton method. This project complements other solvers (Finite Difference, Numerov) by demonstrating a fast root-finding approach and including plots, error analysis and runtime measurement.

---

## Table of contents

- [Overview](#overview)  
- [Features](#features)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Typical output](#typical-output)  
- [How it works (brief)](#how-it-works-brief)  
- [Project structure](#project-structure)  
- [Contributing](#contributing)  
- [License](#license)

---

## Overview

This repository implements the Newton–Raphson (Newton) method to locate eigenvalues (dimensionless `alpha` parameters) for a finite rectangular quantum well. The code applies parity boundary conditions (even / odd) and constructs normalized wavefunctions for visualization. It is ideal for demonstrating how root-finding techniques from numerical analysis can be applied to computational quantum mechanics.

---

## Features

- Uses **SciPy**'s `newton` routine with numerical derivatives for fast convergence.  
- Separately solves for **even** and **odd** parity states.  
- Produces normalized wavefunction plots for the found eigenstates.  
- Compares numerical results with theoretical eigenvalues (when available).  
- Measures execution time for performance comparison.  
- Clean, documented, and easy-to-run single-file script.

---

## Requirements

- Python 3.8+  
- NumPy  
- Matplotlib  
- SciPy

Install dependencies with:

```bash
pip install numpy matplotlib scipy
