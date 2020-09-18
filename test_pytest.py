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


@pytest.mark.parametrize(
    'initials, t0, t1, derived_function, expected, absolute_error, relative_error, expected_error_runge_kutta',
    [
        (numpy.array([cltypes.make_double4(0.0, 0.0, 0.0, 0.0)]), 0.0, 1.0, '1.0, 1.0, 1.0, 1.0',
         numpy.array([cltypes.make_double4(1.0, 1.0, 1.0, 1.0)]), 1e-18, 1e-18, numpy.array([numpy.double(0.0)])),
        (numpy.array([cltypes.make_double4(1.0, 1.0, 1.0, 1.0)]), 0.0, 1.0, '1.0, 1.0, 1.0, 1.0',
         numpy.array([cltypes.make_double4(2.0, 2.0, 2.0, 2.0)]), 1e-18, 1e-18, numpy.array([numpy.double(0.0)])),
        (numpy.array([cltypes.make_double4(1.0, 1.0, 1.0, 1.0)]), 0.0, 1.0, '1.0, -1.0, 1.0, -1.0',
         numpy.array([cltypes.make_double4(2.0, 0.0, 2.0, 0.0)]), 1e-18, 1e-18, numpy.array([numpy.double(0.0)])),
        (numpy.array([cltypes.make_double4(0.0, 1.0, 0.0, 1.0)]), 0.0, 1e-2, 'Y->z, -Y->w, Y->x, -Y->y',
         numpy.array([cltypes.make_double4(numpy.sin(1e-2), numpy.cos(1e-2), numpy.sin(1e-2), numpy.cos(1e-2))]), 2e-2, 2e-10, numpy.array([numpy.double(1e-11)])),
    ]
)
def test_RungeKutta(initials, t0, t1, derived_function, expected, absolute_error, relative_error, expected_error_runge_kutta):
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
        expected_error_runge_kutta, abs=absolute_error)

    numpy.testing.assert_allclose(numpy.array(y.get()[0].tolist()), numpy.array(
        expected[0].tolist()), rtol=relative_error, atol=absolute_error)
