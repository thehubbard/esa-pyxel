#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#

from pympler.asizeof import asizeof
import typing as t

def get_size(obj: t.Any) -> int:
    """Recursively calculates object size in bytes using Pympler library.

    Parameters
    ----------
    obj
        Any input object.
    Returns
    -------
    int
        Object size in bytes.
    """
    return asizeof(obj)


def memory_usage_details(obj: t.Any, print_result: bool = True, human_readable: bool = True, *kw: str) -> dict:

    usage = {}

    for k, v in obj.__dict__.items():
        if hasattr(v, "nbytes") and k in kw:
            usage.update({k.replace("_",""), v.get_size()})
        else:
            pass

    return usage