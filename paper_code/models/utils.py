try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def cycle(iterable):
    '''
    Itertools.cycle tries to save all inputs, which causes memory usage to grow. 
    See: https://github.com/pytorch/pytorch/issues/23900
    '''
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

