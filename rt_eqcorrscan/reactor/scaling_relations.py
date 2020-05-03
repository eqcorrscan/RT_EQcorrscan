"""
Scaling relationships for estimating regions around earthquakes of interest.

Users can select a particular scaling relationship, set the default, or
add their own relationships using context-managers.
"""

import copy
import logging

from typing import Union, Callable

Logger = logging.getLogger(__name__)


SCALING_RELATIONS = {}  # Cache of scaling relations


# ------------------ Context manager for switching out default
class _Context:
    """ class for permanently or temporarily changing items in a dict """

    def __init__(self, cache, value_to_switch):
        """
        :type cache: dict
        :param cache: A dict to store values in
        :type value_to_switch: str
        :param value_to_switch:
            The key in cache to switch based on different contexts
        """
        self.cache = cache
        self.value_to_switch = value_to_switch
        self.previous_value = None

    def __call__(self, new_value, *args, **kwargs):
        """
        Set a new value for the default scaling relationship function.

        This function can be called directly to permanently change the
        default scaling relationship or it may be used as a context manager
        to only modify it temporarily.

        :param new_value:
        :return:
        """
        self.previous_value = copy.deepcopy(
            self.cache.get(self.value_to_switch))
        self.cache[self.value_to_switch] = get_scaling_relation(new_value)
        return self

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.revert()

    def __repr__(self):
        """ this hides the fact _Context instance are returned after calls """
        name = self.cache[self.value_to_switch].__str__()
        if hasattr(self.cache[self.value_to_switch], '__name__'):
            name = self.cache[self.value_to_switch].__name__
        out_str = ("%s changed to %s" % (self.value_to_switch, name))
        return out_str

    def revert(self):
        """ revert the default scaling-relationship to previous value """
        # Have to use the previous value as this may contain some custom
        # stream_xcorr functions
        self.cache[self.value_to_switch] = self.previous_value


set_scaling_relation = _Context(SCALING_RELATIONS, 'default')


def register_scaling_relation(
    name: Union[str, Callable],
    func: Callable = None,
    is_default: bool = False
):

    def wrapper(func, func_name=None):
        fname = func_name or name.__name__ if callable(name) else str(name)
        SCALING_RELATIONS[fname] = func
        func.registered = True
        if is_default:
            SCALING_RELATIONS['default'] = copy.deepcopy(func)
        return func

    # Used as decorator
    if callable(name):
        return wrapper(name)

    # Used as a normal function
    if callable(func):
        return wrapper(func, func_name=name)

    # Called, then used as a decorator
    return wrapper


def get_scaling_relation(name_or_func: Union[str, Callable] = None) -> Callable:
    if callable(name_or_func):
        func = register_scaling_relation(name_or_func)
    else:
        func = SCALING_RELATIONS[name_or_func or 'default']
    assert callable(func), f"{func} is not Callable"

    return func

# ----------------- Define scaling relationships


@register_scaling_relation('wells_coppersmith_surface')
def wells_coppersmith_surface(magnitude: float) -> float:
    """
    Surface rupture length from Wells and Coppersmith 1994, Figure 9.

    Parameters
    ----------
    magnitude:
        Earthquake magnitude

    Returns
    -------
    length:
        Rupture length in km
    """
    return 10 ** ((magnitude - 5.08) / 1.16)


@register_scaling_relation('wells_coppersmith_subsurface', is_default=True)
def wells_coppersmith_subsurface(magnitude: float) -> float:
    """
    Sub-Surface rupture length from Wells and Coppersmith 1994, Figure 14.

    Parameters
    ----------
    magnitude:
        Earthquake magnitude

    Returns
    -------
    length:
        Rupture length in km
    """
    return 10 ** ((magnitude - 4.38) / 1.49)


# Dictionary of functions defined herein, to distinguish from user-defined
SCALING_RELATIONS_ORIGINAL = copy.copy(SCALING_RELATIONS)


if __name__ == "__main__":
    import doctest

    doctest.testmod()