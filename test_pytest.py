import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.cltypes as cltypes
import numpy
import pytest

import rk_pd_4d


def test_parameters_size():
    assert rk_pd_4d.make_initializing_parameters().size == (1024 * 16)


@pytest.mark.parametrize(
    'initials, expected',
    [
        (numpy.array([cltypes.make_double4(0.0, 1.0, 1.0, 0.0)]),
         numpy.array([cltypes.make_double4(1.0, 0.0, 0.0, 1.0)])),
        (numpy.array([cltypes.make_double4(3.0, 4.0, 7.0, -8.0)]),
         numpy.array([cltypes.make_double4(7.0, -8.0, 3.0 / 125.0, 4.0 / 125.0)]))
    ]
)
def test_derivedFn(initials, expected):
    sut = cl.elementwise.ElementwiseKernel(
        rk_pd_4d.context, 'double4 *k, double4 *y, double t', '{}{}'.format(rk_pd_4d.derivedFn, 'derivedFn(k[0], (y[0]), (t))'), 'sut')

    y = cl_array.to_device(rk_pd_4d.queue, initials)
    k = cl_array.empty_like(y)

    sut(k, y, 0.0)

    assert expected == k.get()


@pytest.mark.parametrize(
    'initials, t0, t1, derived_function, expected, expected_error_runge_kutta',
    [
        (numpy.array([cltypes.make_double4(0.0, 0.0, 0.0, 0.0)]), 0.0, 1.0, '1.0, 1.0, 1.0, 1.0',
         numpy.array([cltypes.make_double4(1.0, 1.0, 1.0, 1.0)]), numpy.array([numpy.double(0.0)])),
        (numpy.array([cltypes.make_double4(1.0, 1.0, 1.0, 1.0)]), 0.0, 1.0, '1.0, 1.0, 1.0, 1.0',
         numpy.array([cltypes.make_double4(2.0, 2.0, 2.0, 2.0)]), numpy.array([numpy.double(0.0)])),
        (numpy.array([cltypes.make_double4(1.0, 1.0, 1.0, 1.0)]), 0.0, 1.0, '1.0, -1.0, 1.0, -1.0',
         numpy.array([cltypes.make_double4(2.0, 0.0, 2.0, 0.0)]), numpy.array([numpy.double(0.0)]))
    ]
)
def test_RungeKutta(initials, t0, t1, derived_function, expected, expected_error_runge_kutta):
    sut_derivedFn = '''
        # define derivedFn(k, Y, _t) \
        do \
        {{ \
            k = (double4)({}); \
        }} \
        while (0)

    '''.format(derived_function)

    sut = cl.elementwise.ElementwiseKernel(
        rk_pd_4d.context, 'double4 *y, double4 *y0, double t, double dt, double* error_runge_kutta', '{}{}{}'.format(sut_derivedFn, rk_pd_4d.rungeKutta, 'RungeKutta(y[0], y0[0], t, dt, *error_runge_kutta)'), 'sut')

    y0 = cl_array.to_device(rk_pd_4d.queue, initials)
    y = cl_array.empty_like(y0)
    error_runge_kutta = cl_array.to_device(
        rk_pd_4d.queue, numpy.array([numpy.double(0.0)]))

    sut(y, y0, t0, t1, error_runge_kutta)

    assert expected_error_runge_kutta == error_runge_kutta.get()
    assert expected == y.get()
