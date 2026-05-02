# PINN-based Optimal Pulse Control for a 4-Level Quantum System

A Physics-Informed Neural Network (PINN) with SIREN architecture that simultaneously learns the density matrix dynamics and optimal Gaussian pulse parameters for a 4-level open quantum system governed by the Lindblad master equation.

## Overview

The network approximates the solution to the Lindblad equation:

$$\dot{\rho} = -i[H(t), \rho] + \mathcal{L}[\rho]$$

where the Hamiltonian $H(t)$ is driven by four Gaussian laser pulses with learnable amplitudes $\Omega_k$, center times $t_k$, widths $\sigma_k$, and detunings $\Delta_k$. The goal is to find pulse parameters that return the population to the ground state $\rho_{11}(T) \approx 1$ after time $T = 100$.

## Method

- **Architecture:** SIREN (Sinusoidal Representation Network) with 3 hidden layers of 512 neurons
- **Input:** time $t$ concatenated with the 28 physical parameters
- **Output:** 32-component real vector (real and imaginary parts of $\rho$)
- **Training:** alternating optimization of network weights (Adam, lr=1e-3) and pulse parameters (Adam, lr=5e-3), with curriculum learning over the time horizon
- **Loss:** weighted combination of ODE residual, initial condition, trace preservation, Hermiticity, positivity (coherence bounds), population regularization, and final-state target
- **Validation:** reference solution via `Vern9` ODE solver with tolerances `1e-8`

## Requirements

Julia 1.9+ with the following packages:

```
Lux
Optimisers
Zygote
ComponentArrays
OrdinaryDiffEq
ChainRulesCore
LinearAlgebra
Statistics
Random
Plots
Printf
CSV
DataFrames
```

Install all dependencies by activating the project environment:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Usage

```julia
julia main.jl
```

The script will:
1. Train the PINN for 15 000 iterations with alternating phase optimization
2. Fine-tune the network weights for 10 000 iterations at fixed pulse parameters
3. Validate the learned pulses against a reference ODE solution
4. Save outputs (see below)

## Outputs

| File | Description |
|------|-------------|
| `loss_log.csv` | Training loss components every 10 iterations |
| `finetune_log.csv` | Fine-tuning loss components every 10 iterations |
| `result_fnal.png` | Three-panel plot: learned pulses, PINN solution, ODE reference |

## Physical Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| $\Delta_k$ | $[-10, 10]$ | Detunings (4 levels) |
| $\Omega_k$ | $[1, 40]$ | Pulse amplitudes (8 pulses) |
| $t_k$ | $[30, 70]$ | Pulse center times (8 pulses) |
| $\sigma_k$ | $[2, 6]$ | Pulse widths (8 pulses) |

Decay rates are fixed: $\Gamma_{21} = \Gamma_{31} = \Gamma_{42} = \Gamma_{43} = 0.5$.

## Citation

If you use this code, please cite:

```
[your citation here]
```

This implementation is derived from the PINN methodology described in:

> Rackauckas, C. et al. (2019). DifferentialEquations.jl – A Performant and Feature-Rich Ecosystem for Solving Differential Equations in Julia. *Journal of Open Research Software*.

> Sitzmann, V. et al. (2020). Implicit Neural Representations with Periodic Activation Functions. *NeurIPS*.

## License

[Specify your license, e.g. MIT]