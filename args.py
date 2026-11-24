import torch


class Args:
    # Set to False to use the CPU even when CUDA is available.
    cuda = torch.cuda.is_available()

    n = 40
    stride = 1
    # Constructor arguments retained for compatibility with AFD_NET.
    s = 3
    channel = 3
    num_block = 4
