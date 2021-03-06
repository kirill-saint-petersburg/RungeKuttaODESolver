# Use OpenCL For Prince Dormand Calculations (Using PyOpenCL Arrays and Elementwise)

rungeKutta = '''
inline double RungeKutta(double4* y_next_time, double4* p_y_current_time, double current_time, double time_delta)
{
    const double two_thirds = 2.0 / 3.0;
    const double one_seventwoninths = 1.0 / 729.0;
    const double one_twoninesevenzero = 1.0 / 2970.0;
    const double one_twofivetwozero = 1.0 / 2520.0;
    const double one_fiveninefourzero = 1.0 / 5940.0;

    double4 k1, k2, k3, k4, k5, k6;
    double4 y_t_runge_kutta;

    double h5 = 0.2 * time_delta;

    k1 = derivedFn(p_y_current_time, current_time);                           y_t_runge_kutta = *p_y_current_time + h5 * k1;
    k2 = derivedFn(&y_t_runge_kutta, current_time + h5);                      y_t_runge_kutta = *p_y_current_time + time_delta * (0.075 * k1 + 0.2250 * k2);
    k3 = derivedFn(&y_t_runge_kutta, current_time + 0.3 * time_delta);        y_t_runge_kutta = *p_y_current_time + time_delta * (0.300 * k1 - 0.9000 * k2 + 1.2000 * k3);
    k4 = derivedFn(&y_t_runge_kutta, current_time + 0.6 * time_delta);        y_t_runge_kutta = *p_y_current_time + time_delta * (226.0 * k1 - 675.00 * k2 + 880.00 * k3 + 55.0000 * k4) * one_seventwoninths;
    k5 = derivedFn(&y_t_runge_kutta, current_time + two_thirds * time_delta); y_t_runge_kutta = *p_y_current_time + time_delta * (-1991 * k1 + 7425.0 * k2 - 2660.0 * k3 - 10010.0 * k4 + 10206.0 * k5) * one_twoninesevenzero;
    k6 = derivedFn(&y_t_runge_kutta, current_time + time_delta);

    *y_next_time = *p_y_current_time + time_delta * (341.0 * k1 + 3800.0 * k3 - 7975.0 * k4 + 9477.0 * k5 + 297.0 * k6) * one_fiveninefourzero;

    return length(one_twofivetwozero * (77.0 * k1 - 400.0 * k3 + 1925.0 * k4 - 1701.0 * k5 + 99.0 * k6));
}
'''

princeDormand = '''
# define pd_attempts (32)
# define pd_min_scale_factor (0.0625)
# define pd_max_scale_factor (16.0)

# define pd_success (0)
# define pd_failed_too_many_function_calls (-1)
# define pd_failed_wrong_step (-2)

inline int PrinceDormand(__global double4* p_y_next_time, double4* p_y_current_time, double current_time, double t_next_time, double* p_updated_time_delta, double tolerance)
{
    double tail = t_next_time - current_time;

    if (tail < 0 || *p_updated_time_delta <= 0.0)
    {
        return pd_failed_wrong_step;
    }

    if (tail == 0)
    {
        return pd_success;
    }

    double4 temp_y;
    double4 temp_yt;

    double temp_time_delta = *p_updated_time_delta;

     double temp_tolerance = tolerance / tail;

    double pd_current_time = current_time;

    *p_y_next_time = *p_y_current_time;

    temp_y = *p_y_current_time;

    bool last_interval = false;

    int pd_result = pd_success;

    do
    {
        if (temp_time_delta > tail) { last_interval = true; temp_time_delta = tail; } else if (1.5 * temp_time_delta > tail) temp_time_delta *= 0.5;

        double scale = 1.0;

        int attempts_left = pd_attempts;

        while (--attempts_left >= 0)
        {
            double pd_rk_error = RungeKutta(&temp_yt, &temp_y, pd_current_time, temp_time_delta);

            if (pd_rk_error == 0.0)
            {
                scale = pd_max_scale_factor;

                break;
            }

            double function_module = length(temp_y);

            double function_tolerance = (function_module == 0.0) ? temp_tolerance : function_module;

            scale = min(max(0.8 * sqrt(sqrt(temp_tolerance * function_tolerance / pd_rk_error)), pd_min_scale_factor), pd_max_scale_factor);

            if (pd_rk_error < (temp_tolerance * function_tolerance)) break;

            temp_time_delta *= scale;

            if (temp_time_delta > tail) { temp_time_delta = tail; } else if (1.5 * temp_time_delta > tail) temp_time_delta *= 0.5;
        }

        if (attempts_left <= 0)
        {
                *p_updated_time_delta = temp_time_delta * scale;

                pd_result = pd_failed_too_many_function_calls;

                break;
        }

        temp_y = temp_yt;

        pd_current_time += temp_time_delta; temp_time_delta *= scale; *p_updated_time_delta = temp_time_delta;
    }
    while ((tail = t_next_time - pd_current_time) > 0.0 && !last_interval && (pd_result == pd_success));

    if (pd_result == pd_success)
    {
        *p_y_next_time = temp_yt;
    }

    return pd_result;
}
'''

ode_arguments = 'double t0, double t, double dt, double minimum_dt, int mesh_size, double tolerance, double4 *initializing_parameters, double4 *y, int *pd_result'

ode_operation = '''
    pd_result[i] = pd_success;
    double updated_time_delta = dt;

    y[i] = MakeInitialValues(&initializing_parameters[i]);

    double mesh_time_interval = (t - t0) / mesh_size;

    for (int mesh_index = 0; mesh_index < mesh_size; mesh_index++)
    {
        double4 y_initial = y[i];

        double mesh_t0 = t0 + mesh_index * mesh_time_interval;
        double mesh_t  = mesh_t0 + mesh_time_interval;

        pd_result[i] = PrinceDormand(&y[i], &y_initial, mesh_t0, mesh_t, &updated_time_delta, tolerance);

        if (pd_result[i] != pd_success) break;

        if (updated_time_delta < minimum_dt) updated_time_delta = minimum_dt;
    }
    '''

derivedFnCoulomb = '''
inline double4 derivedFn(double4* Y, double t)
{
    double r = length(Y->xy);

    return (double4)(Y->zw, Y->xy / (r * r * r));
}
'''

makeInitialValuesFromInitializingParametersCoulomb = '''
inline double4 MakeInitialValues(__global double4 *initializing_parameter)
{
    return *initializing_parameter;
}
'''


def make_initializing_parameters_сoulomb():
    import numpy  # Import Numpy number tools
    import pyopencl.cltypes as cltypes

    task_size = 1024 * 16
    x0 = -2048
    v_x0 = 1.0
    return numpy.array(
        [cltypes.make_double4(
            x0,
            0.05 + (x / task_size) * 4.05,
            v_x0,
            0.0
        ) for x in range(task_size)], dtype=cltypes.double4)


if __name__ == '__main__':
    import pyopencl as cl  # Import the OpenCL GPU computing API
    # Import PyOpenCL Array (a Numpy array plus an OpenCL buffer object)
    import pyopencl.array as cl_array

    import numpy  # Import Numpy number tools

    from time import strftime, localtime

    def print_platform_info(platform):
        print('=' * 60)
        print('Platform - Name:     ' + platform.name)
        print('Platform - Vendor:   ' + platform.vendor)
        print('Platform - Version:  ' + platform.version)
        print('Platform - Profile:  ' + platform.profile)
        print('=' * 60)
        print('\n')

    platform = next(platform for platform in cl.get_platforms()
                    if platform.name == 'AMD Accelerated Parallel Processing')

    print_platform_info(platform)

    device = platform.get_devices()

    context = cl.Context(device)  # Initialize the Context
    queue = cl.CommandQueue(context)  # Instantiate a Queue

    initializing_parameters = make_initializing_parameters_сoulomb()

    kernel_side_initializing_parameters = cl_array.to_device(
        queue, initializing_parameters)
    # Create an empty pyopencl destination array
    y = cl_array.empty_like(kernel_side_initializing_parameters)
    pd_result = cl_array.to_device(queue, numpy.zeros(
        initializing_parameters.size, dtype=numpy.int32))

    # Create an elementwise kernel object
    #  - Arguments: a string formatted as a C argument list
    #  - Operation: a snippet of C that carries out the desired map operatino
    #  - Name: the fuction name as which the kernel is compiled

    ode = cl.elementwise.ElementwiseKernel(
        context, ode_arguments, ode_operation,
        name='ode',
        preamble=f'{makeInitialValuesFromInitializingParametersCoulomb}{derivedFnCoulomb}{rungeKutta}{princeDormand}')

    # Call the elementwise kernel
    ode(0.0, 4096.0, 409.6, 0.4096, 32, 1.0e-14,
        kernel_side_initializing_parameters, y, pd_result)

    with open('result_%s.out' % strftime('%d_%m_%Y_%H_%M_%S', localtime()), 'a+') as f_handle:
        numpy.savetxt(f_handle, y.get())
