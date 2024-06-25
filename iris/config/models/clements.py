from collections import defaultdict

import photontorch as pt

from functools import partial
from .factories import(
    _mzi_factory,
    _buffer_wg_factory,
)
from .helper import(
    _NonLinearArray,
    _MixingNonLinearArrayClements,
    _MZIPhaseArray,
    _MixingMZIPhaseArrayClements,
)


class _Capacity2ClementsNxN(pt.Network):
    r""" Helper network for MZIClementsNxN::

        <- cap==2 ->
        0__  ______0
           \/
        1__/\__  __1
               \/
        2__  __/\__2
           \/
        3__/\______3

    """

    def __init__(
        self,
        N=2,
        wg_factory=_buffer_wg_factory,
        mzi_factory=_mzi_factory,
        name=None,
        buffer_wg=False,
        phases=None,
    ):
        """
        Args:
            N (int): number of input waveguides (= number of output waveguides)
            wg_factory (callable): function without arguments which creates the
                waveguides.
            mzi_factory (callable): function without arguments which creates the
                MZIs or any other general 4-port component with  ports defined
                anti-clockwise.
            name (optional, str): name of the component
        """
        num_mzis = N - 1

        if not phases:
            phases = defaultdict(lambda: defaultdict(lambda: None))

        # define components
        components = {}
        for i in range(num_mzis):
            if phases:
                components["mzi%i" % i] = mzi_factory(
                                            phi=phases["mzi%i" % i]['phi'],
                                            theta=phases["mzi%i" % i]['theta'],
                                        )
            else:
                components["mzi%i" % i] = mzi_factory()

        if buffer_wg:
            components["wg0"] = components["wg1"] = _buffer_wg_factory()
        else:
            if phases:
                components["wg0"] = wg_factory(phase=phases["wg0"]['phase'])
                components["wg1"] = wg_factory(phase=phases["wg1"]['phase'])
            else:
                components["wg0"] = components["wg1"] = wg_factory()

        # connections between mzis:
        connections = []
        connections += ["mzi0:1:wg0:0"]
        for i in range(1, num_mzis - 1, 2):
            connections += ["mzi%i:2:mzi%i:0" % ((i - 1), i)]
            connections += ["mzi%i:3:mzi%i:1" % (i, (i + 1))]
        if num_mzis > 1 and N % 2:
            connections += ["mzi%i:2:mzi%i:0" % (num_mzis - 2, num_mzis - 1)]
        if N % 2:
            connections += ["wg1:1:mzi%i:3" % (N - 2)]
        else:
            connections += ["mzi%i:2:wg1:0" % (N - 2)]

        # input connections:
        for i in range(0, num_mzis, 2):
            connections += ["mzi%i:0:%i" % (i, i)]
            connections += ["mzi%i:3:%i" % (i, i + 1)]
        if N % 2:
            connections += ["wg1:0:%i" % (N - 1)]

        # output connections:
        k = i + 2 + N % 2
        connections += ["wg0:1:%i" % k]
        for i in range(1, num_mzis, 2):
            connections += ["mzi%i:1:%i" % (i, k + i)]
            connections += ["mzi%i:2:%i" % (i, k + i + 1)]
        if N % 2 == 0:
            connections += ["wg1:1:%i" % (2 * N - 1)]

        # initialize network
        super(_Capacity2ClementsNxN, self).__init__(components, connections, name=name)


class MZIClementsNxN(pt.Network):
    r""" A unitary matrix network based on the Clements architecture and the 
    photontorch ClementsNxN class, but with MZIs at the place of phaseshifters
    at the end of the mesh.

    Network::

         <--- capacity --->
        0__  ______  ______  _0
           \/      \/      \/
        1__/\__  __/\__  __  _1
               \/      \/  \/
        2__  __/\__  __/\__  _2
           \/      \/      \/
        3__/\______/\______  _3
                           \/
        with:

           3__  __2
              \/    =  MZI
           0__/\__1

    """

    def __init__(
        self,
        N=2,
        capacity=None,
        mzi_factory=_mzi_factory,
        name=None,
    ):
        """
        Args:
            N (int): number of input / output ports (the network represents an
                NxN matrix)
            capacity (int): number of consecutive MZI layers (to span the full
                unitary space one needs capacity >=N).
            mzi_factory (callable): function without arguments which creates
                the MZIs or any other general 4-port component with  ports
                defined anti-clockwise.
            name (optional, str): the name of the network (default: lowercase
                classname)
        """

        if capacity is None:
            capacity = N

        self.N = N
        self.capacity = capacity

        # create components
        components = {}
        for i in range(capacity // 2):
            components["layer%i" % i] = _Capacity2ClementsNxN(
                N=N,
                mzi_factory=mzi_factory,
                wg_factory=_buffer_wg_factory,
                buffer_wg=True,
            )
        if capacity % 2 == 0:
            components["layer%i" % (capacity // 2)] = _MZIPhaseArray(
                self.N, mzi_factory=mzi_factory
            )
        else:
            components["layer%i" % (capacity // 2)] = _MixingMZIPhaseArrayClements(
                self.N,
                mzi_factory=mzi_factory,
                wg_factory=_buffer_wg_factory,
                buffer_wg=True,
            )

        # create connections
        connections = []
        for i in range(capacity // 2):
            for j in range(N):
                connections += ["layer%i:%i:layer%i:%i" % (i, N + j, i + 1, j)]

        # initialize network
        super(MZIClementsNxN, self).__init__(components, connections, name=name)


class NonLinearClementsNxN(pt.Network):
    r""" A unitary matrix network based on the Clements architecture and the 
    photontorch ClementsNxN class, but with ONL at the place of phaseshifters
    at the end of the mesh.

    Network::

         <--- capacity --->
        0__  ______  ______|>__0
           \/      \/
        1__/\__  __/\__  __|>__1
               \/      \/
        2__  __/\__  __/\__|>__2
           \/      \/
        3__/\______/\______|>__3

        with:

           0__|>__1 = Optical Non Linearity

           3__  __2
              \/    =  MZI
           0__/\__1
    """
    def __init__(
        self,
        N=2,
        capacity=None,
        wg_factory=_buffer_wg_factory,
        mzi_factory=_mzi_factory,
        non_linear_factory=lambda : pt.BaseSoa(),
        name=None,
        buffer_wg=False
    ):
        """
        Args:
            N (int): number of input / output ports (the network represents an
                NxN matrix)
            capacity (int): number of consecutive MZI layers (to span the full
                unitary space one needs capacity >=N).
            wg_factory (callable): function without arguments which creates the
                waveguides.
            mzi_factory (callable): function without arguments which creates
                the MZIs or any other general 4-port component with  ports
                defined anti-clockwise.
            name (optional, str): the name of the network (default: lowercase
                classname)
        """
        if capacity is None:
            capacity = N

        self.N = N
        self.capacity = capacity

        # create components
        components = {}
        for i in range(capacity // 2):
            components["layer%i" % i] = _Capacity2ClementsNxN(
                N=N,
                mzi_factory=mzi_factory,
                wg_factory=wg_factory,
                buffer_wg=buffer_wg,
            )

        if capacity % 2 == 0:
            components["layer%i" % (capacity // 2)] = _NonLinearArray(
                self.N, non_linear_factory=non_linear_factory
            )
        else:
            components["layer%i" % (capacity // 2)] = _MixingNonLinearArrayClements(
                self.N,
                mzi_factory=mzi_factory,
                non_linear_factory=non_linear_factory,
                buffer_wg=buffer_wg,
            )

        # create connections
        connections = []
        for i in range(capacity // 2):
            for j in range(N):
                connections += ["layer%i:%i:layer%i:%i" % (i, N + j, i + 1, j)]

        # initialize network
        super(NonLinearClementsNxN, self).__init__(components, connections, name=name)

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
        ret = super(NonLinearClementsNxN, self).terminate(term)
        ret.to(self.device)
        return ret