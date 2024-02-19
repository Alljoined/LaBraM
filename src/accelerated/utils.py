from contextlib import contextmanager

from accelerate import init_empty_weights


@contextmanager
def cond_iew(condition):
    """
    Context that calls init_empty_weights if some condition is met (i.e. condition can be that a pretrained model is present)
    init_empty_weights speeds up large model instantiation when a pretrained model is going to be loaded
    """
    if condition:
        with init_empty_weights():
            yield
    else:
        yield


def get_input_chans(ch_names: str):
    # TODO: This is a placeholder for now
    """
    Get the input channels from the channel names
    """
    return [ch for ch in ch_names if "eeg" in ch]
