import numpy as np

from ..utils.math import create_shared


class HierarData(object):

    def __init__(self, data, limit=None):
        self._cpu_data = data

        if limit is None or limit > data.shape[0]:
            limit = data.shape[0]
        self._limit = limit

        self._gpu_data = create_shared(self._cpu_data[0:limit])
        self._cur_irange = create_shared(np.asarray([0, limit]))

    def prepare(self, irange):
        l, r = irange
        cur_l, cur_r = self._cur_irange.get_value(borrow=True)

        if min(l, r - 1) < cur_l or max(l, r - 1) >= cur_r:
            self._cur_irange.set_value(np.asarray([l, l + self._limit]))
            self._gpu_data.set_value(
                self._cpu_data[l:l + self._limit], borrow=True)

    @property
    def cur_irange(self):
        return self._cur_irange

    @property
    def cpu_data(self):
        return self._cpu_data

    @property
    def gpu_data(self):
        return self._gpu_data
