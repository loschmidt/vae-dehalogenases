__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2021/05/02 00:10:00"

from enum import Enum
from math import sqrt

import numpy as np
from scipy.linalg import fractional_matrix_power

"""
    This file implements all functions needed for covariance matrix adaptation evolution strategy (CMA-ES)
    It can support more modes of CMA-ES which will be added for experiments
"""


class Modes(Enum):
    WEIGHTED = 0
    BEST_FIT = 1
    MU_UPDATE = 2


def weights_recombination(xs, n):
    """ Compute weights of n active members """
    fit_i = [x[0] for x in xs[:n]]
    weights = [f / sum(fit_i) for f in fit_i]
    return weights


def update_mean(mean, xs, sigma, n, mode=Modes.WEIGHTED):
    """
    Update mean by samples and its fitness. Apply weighted step size strategy or simple best fit candidate.
    Parameters:
        mean    - current mean
        xs      - sorted array of fitness tuples from the best to worst
        sigma   - stands for step size
        n       - count of members should be weighted

    Weighted intermediate recombination
        mean[i] = m[i-1] + sigma * sum(z*w)
            z - coordinates of sample normalized by mean around [0,0] -> z - mean
            w - weights of each direction determined by its fitness
    """
    assert n <= len(xs)
    if Modes.WEIGHTED == mode:
        # Weighted intermediate recombination
        zs = [x[1] - mean for x in xs[:n]]
        ws = weights_recombination(xs, n)
        new_mean = mean + sigma * sum([w_i * z_i for w_i, z_i in zip(ws, zs)])
    else:
        # simply the best is chosen as mean
        new_mean = xs[0][1]
    return new_mean


def update_pc(mean, xs, cutoff, pc, n):
    """
    Update anisotropic evolution path. Cumulation of steps.
          Decay factor              Normalization factor
    pc = (1 - c_c) * pc + sqrt(1 - (1 - c_c)^2) * sqrt(mu_eff) * Z_sel

    Parameter:
        n - dimensions constant
    """
    c_c = 4 / (n + 3)  # given by the manuscript, we're working at least with 2D space
    ws = weights_recombination(xs, cutoff)
    zs = [x[1] - mean for x in xs[:n]]
    mu_eff = 1 / sum([w ** 2 for w in ws])  # Effective selection mass
    z_sel = sum([w * x for w, x in zip(ws, zs)])  # Z_sel
    new_pc = (1 - c_c) * pc + sqrt(1 - ((1 - c_c) ** 2)) * sqrt(mu_eff) * z_sel
    return new_pc


def update_Cov(C, mean, xs, cutoff, pc, n):
    """
    Update covariance matrix by the cumulation path with mu rank update
    C = (1 - c_cov)C + (c_cov/mu_cov) * pc * pc^T + c_cov * (1 - (1/mu_cov)) * Z_matrix

    mu_eff == c_cov
    Z_matrix = sum(wi * zi * zi^T)
    Parameters
        n - dimensions constant
    """
    c_cov = 2 / n ** 2  # given by the manuscript
    ws = weights_recombination(xs, cutoff)
    mu_cov = 1 / sum([w ** 2 for w in ws])  # Effective selection mass
    # Compute z matrix
    z_i = [x[1] - mean for x in xs[:cutoff]]
    Z = np.zeros((n, n))
    for w, z in zip(ws, z_i):
        z_t = z[np.newaxis]
        Z += w * (z_t.T @ z_t)
    pc_t = pc[np.newaxis]
    return (1 - c_cov) * C + (c_cov / mu_cov) * (pc_t.T @ pc_t) + c_cov * (1 - (1 - mu_cov)) * Z


def update_ps(ps, C, mean, xs, cutoff, n):
    """
    Cumulation for sigma
    p_s = (1 - c_s) * p_s + sqrt(1 - (1 -c_s)^2) * sqrt(mu_eff) * C^(-1/2) * z_sel

    Parameters
        n - dimensions constant
    """
    c_s = 4 / (n + 3)  # given by the manuscript, we're working at least with 2D space
    ws = weights_recombination(xs, cutoff)
    zs = [x[1] - mean for x in xs[:n]]
    mu_eff = 1 / sum([w ** 2 for w in ws])  # Effective selection mass
    z_sel = sum([w * x for w, x in zip(ws, zs)])  # Z_sel
    C_to_half = fractional_matrix_power(C, -0.5)
    return (1 - c_s) * ps + sqrt(1 - ((1 - c_s) ** 2)) * sqrt(mu_eff) * C_to_half * z_sel

def path_length_control(sigma, ps, mean, xs, cutoff, n):
    """
    Update the step size according to parameters computed previously

    Parameters
        n - dimensions constant
    """
    c_s = 4 / (n + 3)  # given by the manuscript, we're working at least with 2D space
    ws = weights_recombination(xs, cutoff)
    mu_eff = 1 / sum([w ** 2 for w in ws])  # Effective selection mass
    d_s = 1 + sqrt(mu_eff / n)
    p_size = np.linalg.norm(ps)
    return sigma * np.exp((c_s/d_s) * ((p_size / sqrt(n)) - 1))