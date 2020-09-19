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
    'initials, t0, t1, derived_function, expected, delta_absolute_error, absolute_error, relative_error, expected_error_runge_kutta',
    [
        (numpy.array([cltypes.make_double4(0.0, 0.0, 0.0, 0.0)]), 0.0, 1.0, '1.0, 1.0, 1.0, 1.0',
         numpy.array([cltypes.make_double4(1.0, 1.0, 1.0, 1.0)]), 1e-18, 1e-18, 1e-18, numpy.array([numpy.double(0.0)])),
        (numpy.array([cltypes.make_double4(0.0, 0.0, 0.0, 0.0)]), 0.0, 1.0, '1.0, 1.0, 2.0 * Y->x, - 2.0 * Y->y',
         numpy.array([cltypes.make_double4(1.0, 1.0, 1.0, - 1.0)]), 2.3e-16, 2.3e-16, 1e-18, numpy.array([numpy.double(0.0)])),
        (numpy.array([cltypes.make_double4(0.0, 0.0, 0.0, 0.0)]), 0.0, 1.0, '1.0, 1.0, 3.0 * Y->x * Y->x, - 3.0 * Y->y * Y->y',
         numpy.array([cltypes.make_double4(1.0, 1.0, 1.0, - 1.0)]), 3.2e-16, 4.5e-16, 1e-18, numpy.array([numpy.double(0.0)])),
        (numpy.array([cltypes.make_double4(0.0, 0.0, 0.0, 0.0)]), 0.0, 1.0, '1.0, 1.0, 4.0 * Y->x * Y->x * Y->x, - 4.0 * Y->y * Y->y * Y->y',
         numpy.array([cltypes.make_double4(1.0, 1.0, 1.0, - 1.0)]), 7.1e-16, 8.9e-16, 1e-18, numpy.array([numpy.double(0.0)])),
        (numpy.array([cltypes.make_double4(1.0, 1.0, 1.0, 1.0)]), 0.0, 1.0, '1.0, 1.0, 1.0, 1.0',
         numpy.array([cltypes.make_double4(2.0, 2.0, 2.0, 2.0)]), 1e-18, 1e-18, 1e-18, numpy.array([numpy.double(0.0)])),
        (numpy.array([cltypes.make_double4(1.0, 1.0, 1.0, 1.0)]), 0.0, 1.0, '1.0, -1.0, 1.0, -1.0',
         numpy.array([cltypes.make_double4(2.0, 0.0, 2.0, 0.0)]), 1e-18, 1e-18, 1e-18, numpy.array([numpy.double(0.0)])),
        (numpy.array([cltypes.make_double4(0.0, 1.0, 1.0, 0.0)]), 0.0, 1e-2, 'Y->z, -Y->w, -Y->x, Y->y',
         numpy.array([cltypes.make_double4(numpy.sin(1e-2), numpy.cos(1e-2), numpy.cos(1e-2), numpy.sin(1e-2))]), 5.0e-13, 7.4e-12, 7.4e-14, numpy.array([numpy.double(1e-11)])),
        (numpy.array([cltypes.make_double4(0.0, 1.0, 1.0, 0.0)]), 0.0, 1e-2, 'Y->zw, - Y->xy / (length(Y->xy) * length(Y->xy) * length(Y->xy))',
         numpy.array([cltypes.make_double4(numpy.sin(1e-2), numpy.cos(1e-2), numpy.cos(1e-2), - numpy.sin(1e-2))]), 5.0e-12, 1.02e-12, 1.0e-16, numpy.array([numpy.double(1e-10)])),
    ]
)
def test_RungeKutta(initials, t0, t1, derived_function, expected, delta_absolute_error, absolute_error, relative_error, expected_error_runge_kutta):
    sut_derivedFn = f'''double4 derivedFn(double4* Y, double t)
    {{
        return (double4)({derived_function});
    }}'''

    sut = cl.elementwise.ElementwiseKernel(
        context,
        'double4 *y, double4 *y0, double t, double dt, double* error_runge_kutta',
        'double4 temp_y = y[0]; double4 temp_y0 = y0[0]; *error_runge_kutta = RungeKutta(&temp_y, &temp_y0, t, dt); y[0] = temp_y',
        name='sut',
        preamble=f'{sut_derivedFn}{rk_pd_4d.rungeKutta}')

    y0 = cl_array.to_device(queue, initials)
    y = cl_array.empty_like(y0)
    error_runge_kutta = cl_array.to_device(
        queue, numpy.array([numpy.double(0.0)]))

    sut(y, y0, t0, t1, error_runge_kutta)

    assert error_runge_kutta.get() == pytest.approx(
        expected_error_runge_kutta, abs=delta_absolute_error)

    numpy.testing.assert_allclose(numpy.array(y.get()[0].tolist()), numpy.array(
        expected[0].tolist()), rtol=relative_error, atol=absolute_error)
