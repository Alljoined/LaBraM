from contextlib import contextmanager
from accelerate import init_empty_weights

@contextmanager
def cond_iew(condition):
    if condition:
        with init_empty_weights():
            yield
    else:
        yield
