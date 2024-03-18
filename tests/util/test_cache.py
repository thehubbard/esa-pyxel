#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


from diskcache import Cache

from pyxel.util import get_cache


def test_get_cache():
    """Check if function 'get_cache' returns always the same 'Cache' object."""
    cache1 = get_cache()
    cache2 = get_cache()

    assert isinstance(cache1, Cache)
    assert cache1 is cache2
