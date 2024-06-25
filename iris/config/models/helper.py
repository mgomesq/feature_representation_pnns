from collections import defaultdict
import photontorch as pt
from .factories import _buffer_wg_factory


class _MZIPhaseArray(pt.Network):
    """ helper network for ClementsNxN """

    def __init__(
        self, N, mzi_factory, name=None, phases=None
    ):
        """
        Args:
            N (int): number of input / output ports (the network represents an NxN matrix)
            wg_factory (callable): function without arguments which creates the waveguides.
            name (optional, str): name of the component
        """
        self.N = int(N + 0.5)
        components = {}
        connections = []

        if not phases:
            phases = defaultdict(lambda :defaultdict(lambda: None))

        for i in range(self.N):
            components["phase_mzi%i" % i] = mzi_factory(
                                                phi=phases["phase_mzi%i" % i]['phi'],
                                                theta=phases["phase_mzi%i" % i]['theta']
                                            )
            for j in range(2):
                components["phase_mzi%i_term%i" % (i, j)] = pt.Term()
                connections += ["phase_mzi%i:%i:phase_mzi%i_term%i:0" % (i, j, i, j)]

        # input/output connections:
        for i in range(self.N):
            connections += ["phase_mzi%i:3:%i" % (i, i)]
            connections += ["phase_mzi%i:2:%i" % (i, self.N + i)]

        super(_MZIPhaseArray, self).__init__(components, connections, name=name)


class _MixingMZIPhaseArrayClements(pt.Network):
    """ helper network for ClementsNxN """

    def __init__(
        self, N, wg_factory, mzi_factory, name=None, buffer_wg=False, phases=None,
    ):
        """
        Args:
            N (int): number of input / output ports (the network represents an NxN matrix)
            wg_factory (callable): function without arguments which creates the waveguides.
            mzi_factory (callable): function without arguments which creates the MZIs or any other general
                4-port component with  ports defined anti-clockwise.
            name (optional, str): name of the component
        """
        self.N = int(N + 0.5)
        num_mzis = self.N // 2
        components = {}
        connections = []

        if not phases:
            phases = defaultdict(lambda :defaultdict(lambda: None))

        for i in range(self.N):
            components["wg%i" % i] = wg_factory()

            components["phase_mzi%i" % i] = mzi_factory(
                                            phi=phases["phase_mzi%i" % i]['phi'],
                                            theta=phases["phase_mzi%i" % i]['theta'],
                                            )

            for j in range(2):
                components["phase_mzi%i_term%i" % (i, j)] = pt.Term()
                connections += ["phase_mzi%i:%i:phase_mzi%i_term%i:0" % (i, j, i, j)]

        for i in range(num_mzis):
            components["mzi%i" % i] = mzi_factory(
                                        phi=phases["mzi%i" % i]['phi'],
                                        theta=phases["mzi%i" % i]['theta'],
                                    )
        if self.N % 2:
            if buffer_wg:
                components["wg_"] = _buffer_wg_factory()
            else:
                components["wg_"] = wg_factory()
  
        for i in range(num_mzis):
            connections += ["mzi%i:1:phase_mzi%i:3" % (i, 2 * i)]
            connections += ["mzi%i:2:phase_mzi%i:3" % (i, 2 * i + 1)]
        if self.N % 2:
            connections += ["wg_:1:phase_mzi%i:3" % (self.N - 1)]

        # input connections:
        for i in range(0, num_mzis):
            connections += ["mzi%i:0:%i" % (i, 2 * i)]
            connections += ["mzi%i:3:%i" % (i, 2 * i + 1)]
        if self.N % 2:
            connections += ["wg_:0:%i" % (self.N - 1)]

        # output connections:
        for i in range(self.N):
            connections += ["phase_mzi%i:2:%i" % (i, self.N + i)]

        super(_MixingMZIPhaseArrayClements, self).__init__(
            components, connections, name=name
        )


class _NonLinearArray(pt.Network):
    """ helper network for ClementsNxN """

    def __init__(
        self, N, non_linear_factory=lambda :pt.BaseSoa(), name=None,
    ):
        """
        Args:
            N (int): number of input / output ports (the network represents an NxN matrix)
            non_linear_factory (callable): function without arguments which creates the Optical Non Linearities.
            name (optional, str): name of the component
        """
        self.N = int(N + 0.5)
        components = {}
        connections = []

        for i in range(self.N):
            components["ONL%i" % i] = non_linear_factory()

        # input/output connections:
        for i in range(self.N):
            connections += ["ONL%i:0:%i" % (i, i)]
            connections += ["ONL%i:1:%i" % (i, self.N + i)]

        super(_NonLinearArray, self).__init__(components, connections, name=name)
        

class _MixingNonLinearArrayClements(pt.Network):
    """ helper network for ClementsNxN """

    def __init__(
        self, N, mzi_factory, non_linear_factory=lambda :pt.BaseSoa(), wg_factory=pt.Waveguide(), name=None, buffer_wg=False, phases=None,
    ):
        """
        Args:
            N (int): number of input / output ports (the network represents an NxN matrix)
            non_linear_factory (callable): function without arguments which creates the Optical Non Linearities.
            mzi_factory (callable): function without arguments which creates the MZIs or any other general
                4-port component with  ports defined anti-clockwise.
            name (optional, str): name of the component
        """
        self.N = int(N + 0.5)
        num_mzis = self.N // 2
        components = {}

        if not phases:
            phases = defaultdict(lambda :defaultdict(lambda :None))

        for i in range(self.N):
            # components["ONL%i" % i] = _buffer_wg_factory()
            components["ONL%i" % i] = non_linear_factory()

        for i in range(num_mzis):
            components["mzi%i" % i] = mzi_factory(
                                        phi=phases["mzi%i" % i]['phi'],
                                        theta=phases["mzi%i" % i]['theta'],
                                    )
        if self.N % 2:
            if buffer_wg:
                components["wg_"] = _buffer_wg_factory()
            else:
                components["wg_"] = wg_factory()
        connections = []
        for i in range(num_mzis):
            connections += ["mzi%i:1:ONL%i:0" % (i, 2 * i)]
            connections += ["mzi%i:2:ONL%i:0" % (i, 2 * i + 1)]
        if self.N % 2:
            connections += ["wg_:1:ONL%i:0" % (self.N - 1)]

        # input connections:
        for i in range(0, num_mzis):
            connections += ["mzi%i:0:%i" % (i, 2 * i)]
            connections += ["mzi%i:3:%i" % (i, 2 * i + 1)]
        if self.N % 2:
            connections += ["wg_:0:%i" % (self.N - 1)]

        # output connections:
        for i in range(self.N):
            connections += ["ONL%i:1:%i" % (i, self.N + i)]

        super(_MixingNonLinearArrayClements, self).__init__(
            components, connections, name=name
        )