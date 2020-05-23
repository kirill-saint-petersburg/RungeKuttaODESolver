import rk_pd_4d

def test_parameters_size():
    assert rk_pd_4d.make_initializing_parameters().size == (1024 * 16)
