"""
EQcorrscan's simple logging.

:copyright:
    EQcorrscan developers.

:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def verbose_print(string, verbosity, print_level):
    """
    Print the string if the print_level exceeds the debug_level.

    :type string: str
    :param string: String to print
    :type print_level: int
    :param print_level: Print-level for statement
    :type verbosity: int
    :param verbosity: Output level for function

    .. rubric:: Example
    >>> verbose_print("Albert", 2, 0)
    >>> verbose_print("Norman", 0, 2)
    Norman
    """
    if print_level > verbosity:
        print(string)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
