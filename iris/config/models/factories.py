import torch
import numpy as np
import photontorch as pt

from config.models.mzi import BoundedMziThermal


def random_init_phase(bounds, N, seed=None):

    def _sigmoid(weights, bounds):
        a, b = bounds
        scaled_data = torch.sigmoid(weights)
        data = (b - a) * scaled_data + a
        return data

    def _xavier_uniform(N, seed=None):
        generator = np.random.default_rng(seed=seed)
        bound = np.sqrt(6/(N+N))
        return generator.uniform(low=-bound, high=bound)

    unbounded_weight = _xavier_uniform(N, seed)
    bounded_weight = _sigmoid(torch.tensor(unbounded_weight), bounds)
    return float(bounded_weight)

def _rnd_thermal_mzi_factory(N=4, seed=42, phi=None, theta=None):
    BASE_LENGTH = 0.
    return BoundedMziThermal(
            phi=random_init_phase((0,2*np.pi), N, seed),
            theta=random_init_phase((0,(np.pi/2)), N, seed),
            trainable=True,
            length=BASE_LENGTH,
            temperature=300.
        )

def _buffer_wg_factory():
    return pt.Waveguide(
        phase=0.0,
        trainable=False,
        length=0.0,
    )


def _mzi_factory(phi=None, theta=None):
    BASE_LENGTH = 0.
    if phi and theta:
        return pt.BoundedMzi(
            phi=phi,
            theta=theta,
            trainable=True,
            length=BASE_LENGTH,
        )
    return pt.BoundedMzi(
        phi=2 * np.pi * 0.5,
        theta= (np.pi/2) * 0.5,
        trainable=True,
        length=BASE_LENGTH,
    )