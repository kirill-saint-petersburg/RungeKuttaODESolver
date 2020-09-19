import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.cltypes as cltypes
import numpy
import pytest

import rk_pd_4d

platform = next(platform for platform in cl.get_platforms()
                if platform.name == 'Intel(R) OpenCL')

device = platform.get_devices()

context = cl.Context(device)  # Initialize the Context
queue = cl.CommandQueue(context)  # Instantiate a Queue


def test_parameters_size():
    assert rk_pd_4d.make_initializing_parameters_сoulomb().size == (1024 * 16)
