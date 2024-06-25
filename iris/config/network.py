from functools import partial

import torch
import photontorch as pt

from config.models.neural import FullyConnectedNN
from config.models.factories import _rnd_thermal_mzi_factory

def test_circuit(circuit, test_dataloader, encoding_function, softmax, loss_function):

    for data, labels in test_dataloader:

        sources = encoding_function(data)
 
        detected = circuit(sources, power = True)[-1,0,:,:] # [t, wl, det, batch]
        detected = detected.swapaxes(0, 1) # make it [batch, det]
        outputs = softmax(detected)

        correct_predictions = 0
        for idx, output in enumerate(outputs):
            predicted_label = torch.argmax(output)
            if predicted_label == labels[idx]:
                correct_predictions += 1

        test_acc = correct_predictions/len(outputs)
        test_loss = loss_function(outputs, labels)

    return test_loss, test_acc


def create_circuit_iris(device='cpu', beta=1, input_bias=False, bias=0.0, seed=42): #IRIS

    if input_bias:
        term = [ #INPUT
            pt.Term(name='t1'),
            pt.Term(name='t2'),
            pt.Source(name="s1"),
            pt.Source(name="s2"),
            pt.Source(name="s3"),
            pt.Term(name='t3'),
        ]

    else: 
        term = [ #INPUT
                    pt.Term(name='t1'),
                    pt.Term(name='t2'),
                    pt.Source(name="s1"),
                    pt.Source(name="s2"),
                    pt.Term(name='t3'),
                    pt.Term(name='t4'),
                ]

    term += [ #OUTPUT
        pt.Term(name='t5'),
        pt.Term(name='t6'),
        pt.Detector(name="d0"),
        pt.Detector(name="d1"),
        pt.Detector(name="d2"),
        pt.Term(name='t7'),
    ]

    circuit = FullyConnectedNN(
                                N=6,
                                layers=2,
                                mzi_factory=partial(_rnd_thermal_mzi_factory, N=6, seed=seed),
                                beta=beta,
                                bias=bias
                            )
    
    circuit = circuit.to(device).terminate(term).initialize()
    circuit.__class__ = pt.Network

    return circuit


def create_circuit_iris_independent(device='cpu', beta=1, input_bias=False, bias=0.0, seed=42): #IRIS


    if input_bias:
        term = [ #INPUT
            pt.Term(name='t1'),
            pt.Source(name="s0"),
            pt.Source(name="s1"),
            pt.Source(name="s2"),
            pt.Source(name="s3"),
            pt.Source(name="s4"),
        ]
    else: 
        term = [ #INPUT
            pt.Term(name='t1'),
            pt.Source(name="s0"),
            pt.Source(name="s1"),
            pt.Source(name="s2"),
            pt.Source(name="s3"),
            pt.Term(name='t2'),
        ]

    term += [ #OUTPUT
        pt.Term(name='t5'),
        pt.Term(name='t6'),
        pt.Detector(name="d0"),
        pt.Detector(name="d1"),
        pt.Detector(name="d2"),
        pt.Term(name='t7'),
    ]

    circuit = FullyConnectedNN(
                        N=6,
                        layers=2,
                        mzi_factory=partial(_rnd_thermal_mzi_factory, N=6, seed=seed),
                        beta=beta,
                        bias=bias,
                    )
    circuit = circuit.to(device).terminate(term).initialize()
    circuit.__class__ = pt.Network

    return circuit
