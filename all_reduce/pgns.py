#! /usr/bin/env python3


import collections
import glob
import os

import pandas


class PGNS:
    def __init__(self, model_name, init_batch_size, validation_dir) -> None:
        self.model_name = model_name
        self.cur_batch_size = init_batch_size

        # init validation
        validation = {}
        for path in glob.glob(os.path.join(f"{validation_dir}/{model_name}/", "*.csv",)):
            bsz = int(path.split("_")[-1].split(".")[0])
            validation[bsz] = pandas.read_csv(path)
        self.validation = collections.OrderedDict(sorted(validation.items()))

    def __validated_batch_sizes(self, batch_size):
        lower_bsz = upper_bsz = None
        for bsz in self.validation:
            if bsz <= batch_size:
                lower_bsz = bsz
            if bsz >= batch_size:
                upper_bsz = bsz
                break
        assert lower_bsz is not None and upper_bsz is not None, "{} {}".format(batch_size, list(self.validation))
        assert lower_bsz <= batch_size <= upper_bsz
        return lower_bsz, upper_bsz

    def __pgns_cal(self, batch_size, epoch):
        lower_bsz, upper_bsz = self.__validated_batch_sizes(batch_size)
        lower_sqr = self.validation[lower_bsz].grad_sqr[epoch]
        upper_sqr = self.validation[upper_bsz].grad_sqr[epoch]
        lower_var = self.validation[lower_bsz].grad_var[epoch]
        upper_var = self.validation[upper_bsz].grad_var[epoch]
        if lower_bsz == upper_bsz:
            assert lower_sqr == upper_sqr and lower_var == upper_var
            return lower_var / lower_sqr
        # Linear interpolation between lower_bsz and upper_bsz.
        sqr = ((batch_size - lower_bsz) * upper_sqr + (upper_bsz - batch_size) * lower_sqr) / (upper_bsz - lower_bsz)
        var = ((batch_size - lower_bsz) * upper_var + (upper_bsz - batch_size) * lower_var) / (upper_bsz - lower_bsz)
        return var / sqr

    def set_cur_bsz(self, batch_size):
        self.cur_batch_size = batch_size

    # return value may not be int
    def get_iter_num(self, batch_size, epoch):
        pgns = self.__pgns_cal(self.cur_batch_size, epoch)
        return 1 + pgns / batch_size
