import torch
import numpy as np

def encoding_function_linear(a, b):
    return a + b*1j


def encoding_function_exponential(a, b):
    return a*torch.exp(b*np.pi*1j)


def get_sources(inputs, input_bias=False, device='cpu'):
    ''' Takes inputs and shapes them to be used in photontorch simulation

    inputs - np.array in the shape of:
    [[psi1, psi2, psi3 ... #complex inputs], ... ] # Batch
    psi -> complex number in form 1 + 1j
    '''

    if input_bias:
        bias_value = 1 + 0j
        num_batches = inputs.shape[0]
        bias_inputs = torch.tensor([[bias_value]] * num_batches, device=device) 
        inputs = torch.cat((inputs, bias_inputs), axis=1)

    inputs = torch.flip(inputs, dims=[1]).to(torch.complex64)
    inputs = [torch.real(inputs), torch.imag(inputs)]
    inputs = torch.stack(inputs).to(device=device)

    sources = torch.tensor(
        inputs.permute(0, 2, 1), # Switching from ['c', 'b', 's'] -> ['c','s','b']
        dtype=torch.get_default_dtype(),
        names=['c','s','b'],
        device=device,
    )

    return sources


def split_in_half(data:torch.Tensor) -> tuple:
    ''' Splits a dataset shaped as [#batch, #inputs] into two in the #inputs
    dimention, to obtain ([#batch, #inputs/2], [#batch, #inputs/2])

    Arguments
    ---------
    data - a torch.Tensor with 2 shaped as [#batch, #inputs]

    Returns
    -------
    Returns a tuple of two halfs of data
    '''

    half_of_inputs = data.shape[1]//2
    split_a, split_b= torch.split(data, half_of_inputs, dim=1)
    return split_a, split_b

def encode_data(data, encoding = 'independent', input_bias=False, device='cpu'):

    if encoding == 'independent':
        sources = get_sources(data, input_bias, device)
        return sources
    
    split_a, split_b = split_in_half(data)

    if encoding == 'linear':
        inputs = encoding_function_linear(split_a, split_b)
    elif encoding == 'exponential':
        inputs = encoding_function_exponential(split_a, split_b)
    elif encoding == 'special':
        # Combines [A,B,C,D] -> [A,C] (linear), [B,D] (exponential)
        
        split_x1, split_x2  =  split_in_half(split_a)
        split_x3, split_x4  =  split_in_half(split_b)

        input_x1_x3 = encoding_function_linear(split_x1, split_x3)
        input_x2_x4 = encoding_function_exponential(split_x2, split_x4)

        inputs = torch.cat((input_x1_x3, input_x2_x4), axis=1)
        
    else:
        raise NotImplementedError('Unkown Encoding')
    
    sources = get_sources(inputs, input_bias, device)
    return sources
