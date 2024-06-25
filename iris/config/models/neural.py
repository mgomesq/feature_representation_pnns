from collections import defaultdict

import torch
import photontorch as pt

from .clements import (
    _Capacity2ClementsNxN,
    _MZIPhaseArray,
    _NonLinearArray,
    _MixingMZIPhaseArrayClements,
    _MixingNonLinearArrayClements,
)

from .factories import(
    _buffer_wg_factory,
)

def softp(x, beta):
    return (1/(beta)) * torch.log(1 + torch.exp(beta*x))


class ActivationFunction(pt.Component):
    """ Softplus activation function component

    Terms::

        0 ---- 1
    """

    num_ports = 3
    def __init__(self, beta=1, bias=0.):
        super(ActivationFunction, self).__init__()
        self.beta = beta
        self.activation = torch.nn.Softplus(beta=beta)
        self.bias = bias

    def action(self, t, x_in, x_out):
        """ Nonlinear action of the component on its active nodes

        Args:
            t (float): the current time in the simulation
            x_in (torch.Tensor[#active nodes, 2, #wavelengths, #batches]): the input tensor
                used to define the action
            x_out (torch.Tensor[#active nodes, 2, #wavelengths, #batches]): the output
                tensor. The result of the action should be stored in the
                elements of this tensor.

        """
        a_in, h, _ = x_in  # unpack the three active nodes

        # input amplitude
        x_out[0] = a_in  # nothing happens to the input active node

        re, im = a_in[:,0,:]

        complex_numbers = re + im*1j
        modulus = torch.abs(complex_numbers)
        angles = (complex_numbers).angle()

        scaler = self.activation(modulus+self.bias)

        re_out = torch.cos(angles)*scaler
        im_out = torch.sin(angles)*scaler

        x_out[2] = torch.stack([torch.stack([re_out,im_out])]).swapaxes(0,1)

    def set_actions_at(self, actions_at):
        actions_at[:] = 1

    def set_S(self, S):
        S[0, :, 0, 0] = 1.0
        S[0, :, 1, 1] = 1.0
        S[0, :, 2, 2] = 1.0
        return S

    def set_C(self, C):
        C[1, 1] = 1.0  # the internal state should be connected onto itself.


class FullyConnectedNN(pt.Network):
    r"""
    TODO Docstring
    """
    def __init__(
        self,
        N=2,
        layers=1,
        beta=1,
        name=None,
        phases=None,
        mzi_factory=None,
        bias=0,
    ):
        """
        Args:
            N (int): number of input / output ports (the network represents an
                NxN matrix)
            name (optional, str): the name of the network (default: lowercase
                classname)
        """

        buffer_wg_factory = _buffer_wg_factory
        
        if beta == None:
            print('using buffer')
            non_linear_factory = _buffer_wg_factory
        else:
            non_linear_factory =  lambda : ActivationFunction(beta=beta, bias=bias)

        self.N = N
        self.capacity = N

        if not phases:
            phases = defaultdict(lambda: None)

        # create components
        components = {}
        self.connections = []

        for L in range(layers):
            for i in range(self.capacity // 2):

                components[f'layer{L}_vsigma_{i}'] = _Capacity2ClementsNxN(
                    N=N,
                    mzi_factory=mzi_factory,
                    wg_factory=buffer_wg_factory,
                    buffer_wg=True,
                    name=f'layer{L}_vsigma_{i}',
                    phases=phases[f'layer{L}_vsigma_{i}']
                )

                components[f'layer{L}_u_onl_{i}'] = _Capacity2ClementsNxN(
                    N=N,
                    mzi_factory=mzi_factory,
                    wg_factory=buffer_wg_factory,
                    buffer_wg=True,
                    name=f'layer{L}_u_onl_{i}',
                    phases=phases[f'layer{L}_u_onl_{i}'],
                )

            # Last layer for each mesh. Best case it is only an array of some
            #   components. Worst case, there should be a mix of MZIs also.
            if self.capacity % 2 == 0:
                components[f'layer{L}_vsigma_{self.capacity // 2}'] = _MZIPhaseArray(
                    self.N,
                    mzi_factory=mzi_factory,
                    name=f'layer{L}_vsigma_{self.capacity // 2}',
                    phases=phases[f'layer{L}_vsigma_{self.capacity // 2}']
                )

                components[f'layer{L}_u_onl_{self.capacity // 2}'] = _NonLinearArray(
                    self.N,
                    non_linear_factory=non_linear_factory,
                    name=f'layer{L}_u_onl_{self.capacity // 2}',
                )

            else:
                components[f'layer{L}_vsigma_{self.capacity // 2}'] = _MixingMZIPhaseArrayClements(
                    self.N,
                    mzi_factory=mzi_factory,
                    wg_factory=buffer_wg_factory,
                    buffer_wg=True,
                    name=f'layer{L}_vsigma_{self.capacity // 2}',
                    phases=phases[f'layer{L}_vsigma_{self.capacity // 2}'],
                )

                components[f'layer{L}_u_onl_{self.capacity // 2}'] = _MixingNonLinearArrayClements(
                    self.N,
                    mzi_factory=mzi_factory,
                    non_linear_factory=non_linear_factory,
                    buffer_wg=True,
                    name=f'layer{L}_u_onl_{self.capacity // 2}',
                    phases=phases[f'layer{L}_u_onl_{self.capacity // 2}'],
                )

            # create inter-mesh connections
            for i in range(self.capacity // 2):
                self.fully_connect(f'layer{L}_vsigma_{i}', f'layer{L}_vsigma_{i+1}')
                self.fully_connect(f'layer{L}_u_onl_{i}', f'layer{L}_u_onl_{i+1}')

            # create intra-mesh connections:: last-layer to first layer
            self.fully_connect(f'layer{L}_vsigma_{self.capacity // 2}', f'layer{L}_u_onl_{0}')

            if L != (layers - 1):
                self.fully_connect(f'layer{L}_u_onl_{self.capacity // 2}', f'layer{L+1}_vsigma_{0}')

        # initialize network
        super(FullyConnectedNN, self).__init__(components, self.connections, name=name)


    def fully_connect(self, A, B):
        for j in range(self.N):
            self.connections += [f"{A}:{self.N+j}:{B}:{j}"]


    def terminate(self, term=None):
        """ Terminate open conections with the term of your choice

        Args:
            term: (Term|list|dict): Which term to use. Defaults to Term. If a
                dictionary or list is specified, then one needs to specify as
                many terms as there are open connections.

        Returns:
            terminated network with sources on the left and detectors on the right.
        """
        if term is None:
            term = [pt.Source(name="s%i" % i) for i in range(self.N)]
            term += [pt.Detector(name="d%i" % i) for i in range(self.N)]
        ret = super(FullyConnectedNN, self).terminate(term)
        ret.to(self.device)
        return ret
