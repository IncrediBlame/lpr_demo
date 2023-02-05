# Debug decorator and constants
# Debug decorator should be replaced with NOOP in production

import functools
from typing import Callable, Any


# debug constants to control debug level
DEBUG_LVL_1 = False
DEBUG_LVL_2 = False
DEBUG_LVL_3 = False
DEBUG_LVL_4 = False
DEBUG_LVL_10 = False

# DEBUG_LVL_1 = True
# DEBUG_LVL_2 = True
# DEBUG_LVL_3 = True
# DEBUG_LVL_4 = True
# DEBUG_LVL_10 = True


def debug(func: Callable) -> Callable:
    """
    Debug decorator. Controls debug levels.
    Should be replaced with NOOP in production.
    """
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs) -> Any:
        # Uncomment line below in production :)
        return

        if 'debug_lvl' not in kwargs:
            raise Exception("Debug functions must specify debug_lvl")
        debug_lvl = kwargs['debug_lvl']

        val = None
        enabled = False
        if DEBUG_LVL_1 and debug_lvl >= 1:
            enabled = True
        if DEBUG_LVL_2 and debug_lvl >= 2:
            enabled = True
        if DEBUG_LVL_3 and debug_lvl >= 3:
            enabled = True
        if DEBUG_LVL_4 and debug_lvl >= 4:
            enabled = True
        if DEBUG_LVL_10 and debug_lvl >= 10:
            enabled = True
        
        if enabled:
            val = func(*args, **kwargs)
        return val

    return wrapper_debug