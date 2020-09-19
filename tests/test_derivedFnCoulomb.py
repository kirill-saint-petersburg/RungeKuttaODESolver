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


@pytest.mark.parametrize(
    'initials, expected',
    [
        (numpy.array([cltypes.make_double4(0.0, 1.0, 1.0, 0.0)]),
         numpy.array([cltypes.make_double4(1.0, 0.0, 0.0, 1.0)])),
        (numpy.array([cltypes.make_double4(3.0, 4.0, 7.0, -8.0)]),
         numpy.array([cltypes.make_double4(7.0, -8.0, 3.0 / 125.0, 4.0 / 125.0)]))
    ]
)
def test_derivedFnCoulomb(initials, expected):
    sut = cl.elementwise.ElementwiseKernel(
        context, 'double4 *k, double4 *y, double t', 'double4 temp_y = y[0]; k[0] = derivedFn(&temp_y, t)', name='sut', preamble=rk_pd_4d.derivedFnCoulomb)

    y = cl_array.to_device(queue, initials)
    k = cl_array.empty_like(y)

    sut(k, y, 0.0)

    assert expected == k.get()
