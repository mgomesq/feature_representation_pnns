""" Realizes the training of several PNNs on the IRIS dataset."""

from itertools import repeat
from functools import partial

import os
import sys
import torch
import numpy as np
import pandas as pd
import photontorch as pt

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.multiprocessing import Pool, set_start_method

from config.encoding import encode_data
from config.dataset import IrisDataset
from config.log import log_this
from config.network import (
    create_circuit_iris,
    create_circuit_iris_independent,
    test_circuit
)

# Simultaion
DT = 3.e-10 #[s]
TOTAL_TIME = 10*DT #[s]

# Multiprocessing
CORE_COUNT = 1 # or cpu_count()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training
EPOCHS = 1
BATCHES = 5
SHUFFLE_BATCH = True
MOMENTUM = 0.9
INITIAL_LR = 1e-2
MIN_LR = 1e-4

# Network definitions
BETA = 1.0
BIAS = 0.0
INPUT_BIAS = True
FEATURE_ORDER = [0, 1, 2, 3]
ENCODING_TYPES = ['linear', 'exponential', 'independent']

# FEATURE_ORDER:
# [0,1,2,3] -> [sepal length, petal length], [sepal width, petal width]
# [0,2,1,3] -> [sepal length, sepal width] , [petal length, petal width]

SEED = None
START_AT = 0
ITERATION_COUNT = 1
ITERATION_RANGE = list(range(START_AT, START_AT + ITERATION_COUNT))

def simulate_and_save(idx, encoding_type):
    ''' Simulates and saves the result of the training of a PNN on the IRIS
    dataset. This was made into a function to spawn several trainings with Pool.
    This function makes use of several global variables defined in this file.

    Arguments
    ---------
    idx - int, reference of the instance being trained.
    encoding_type - str, type of encoding to be used. can be 'exponential', 'linear' 
    or 'independent'.

    Returns
    -------
    None, but it saves an output to the output_folder with training results.

    '''

    log_this(f'IDX {idx}: START')

    try:
        output_folder = f'outputs/{encoding_type}/'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        env = pt.Environment(
                    freqdomain=False,
                    time= np.arange(0, TOTAL_TIME, DT),
                    grad=True,
                )

        with env:

            if encoding_type == 'independent':
                create_circuit = create_circuit_iris_independent
            else:
                create_circuit = create_circuit_iris

            circuit = create_circuit(
                        beta=BETA,
                        bias=BIAS,
                        input_bias=INPUT_BIAS,
                        seed=SEED,
                        device=DEVICE,
                        )

            encoding_function = partial(
                                    encode_data,
                                    encoding=encoding_type,
                                    input_bias=INPUT_BIAS,
                                    device=DEVICE,
                                )

            num_param = len(list(circuit.parameters()))
            log_this(f'IDX {idx}: Circuit created in {circuit.device}\
                        with {num_param} param.')

            ## Dataset Definition ##
            iris_dataset = IrisDataset(feature_order=FEATURE_ORDER, device=DEVICE)

            train_dataloader = DataLoader(
                                iris_dataset.train,
                                batch_size=int(len(iris_dataset.train)/BATCHES),
                                shuffle=SHUFFLE_BATCH,
                                )
            test_dataloader = DataLoader(
                                iris_dataset.test,
                                batch_size=len(iris_dataset.test),
                                )
            log_this(f'IDX {idx}: Dataset OK')


            softmax = torch.nn.Softmax(dim=1) # Alongside Detector dimention
            loss_function = torch.nn.CrossEntropyLoss()

            optimizer = torch.optim.SGD(
                            circuit.parameters(),
                            lr=INITIAL_LR,
                            momentum=MOMENTUM,
                            )

            scheduler = ReduceLROnPlateau(
                            optimizer,
                            'min',
                            factor=0.5,
                            patience=10,
                            cooldown=10,
                            min_lr=MIN_LR,
                            )

            data_df = pd.DataFrame()

            for epoch in range(EPOCHS):

                log_this(f'EPOCH START: {epoch} epoch')

                epoch_loss = []

                for batch, (data, labels) in enumerate(train_dataloader):
                    log_this(f'STARTING: {epoch} {encoding_type} in batch {batch}')
                    sources = encoding_function(data)
                    optimizer.zero_grad()

                    detected = circuit(sources, power = True)[-1,0,:,:] # [t, wl, det, batch]
                    detected = detected.swapaxes(0, 1) # make it [batch, det]
                    outputs = softmax(detected)
                    loss = loss_function(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    epoch_loss.append(loss.item())

                mean_epoch_loss = np.array(epoch_loss).mean()
                scheduler.step(mean_epoch_loss)
                new_lr = optimizer.state_dict()['param_groups'][0]['lr']

                test_loss, test_acc = test_circuit(
                                            circuit,
                                            test_dataloader,
                                            encoding_function,
                                            softmax,
                                            loss_function,
                                        )

                data = {
                    'epoch': epoch,
                    'test_acc': test_acc,
                    'train_loss': mean_epoch_loss,
                    'test_loss': test_loss.item(),
                }

                data_df = pd.concat(
                                [data_df, pd.DataFrame.from_records([data])],
                                axis=0,
                            )

                log_this(f'IDX {idx}: EPOCH {epoch}\
                            \tloss: {mean_epoch_loss:.4f}\
                            \ttest_acc: {test_acc*100:.2f}%\
                            \tlr: {new_lr}')

            data_df.to_csv(output_folder+f'results{idx}.csv', index=False)

    except KeyboardInterrupt:
        print('Interrupted... exiting')
        pool.terminate()
        sys.exit(0)

if __name__ == '__main__':

    log_this(f'starting {ITERATION_COUNT} counts, encoding: {ENCODING_TYPES}')

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    parameter_list = []
    for encoding in ENCODING_TYPES:
        parameter_list += list(zip(ITERATION_RANGE, repeat(encoding)))

    with Pool(CORE_COUNT) as pool:
        log_this(f'starting computations on {CORE_COUNT} cores')
        pool.starmap(simulate_and_save, parameter_list)

    log_this('PROGRAM FINISHED OK')
