import random
from typing import List

import numpy as np

from classifier.Model import INPUT_LENGTH

REGULAR = "reg"
CHIMERIC = "chi"
REPEAT = "rep"
LOW_QUALITY = "loq"


class Sequence:

    def __init__(self):
        self.query_id = "-1"
        self.query_len = "0"
        self.type = REGULAR
        self.bps = np.zeros(0)

    def setup(self, query_id, query_len):
        self.query_id = query_id
        self.query_len = query_len
        self.bps = np.zeros(query_len)

    def print(self, file_handle):
        overlaps = ",".join(map(str, self.bps.tolist()))
        serialized = "%s\t%s\t%s\t%s\n" % (self.query_id, self.query_len, self.type, overlaps)
        file_handle.write(serialized)

    def append(self, query_hit_start, query_hit_end):
        for index in range(query_hit_start, query_hit_end):
            self.bps[index] += 1

    def pad_sequence(self):
        padding = INPUT_LENGTH - int(self.query_len)
        if padding == 0:
            return

        print(padding)
        extended_bps: List = self.bps.tolist()

        i = 0
        while padding > 0:
            current_len = len(extended_bps)

            bps_to_interpolate = min(padding, current_len)
            interpolation_indices = random.sample(population=range(current_len), k=bps_to_interpolate)

            last_index = current_len - 1
            for idx in range(current_len):  # Go through existing BPs
                if idx in interpolation_indices:
                    if idx == last_index:
                        point = (extended_bps[idx - 1] + extended_bps[idx]) // 2
                        extended_bps.insert(idx, point)
                    else:
                        point = (extended_bps[idx] + extended_bps[idx + 1]) // 2
                        extended_bps.insert(idx + 1, point)

            padding = padding - bps_to_interpolate
            i = i + 1

        self.bps = np.array(extended_bps)

    def trim_sequence(self):
        excess = int(self.query_len) - INPUT_LENGTH
        if excess == 0:
            return

        drop_indexes = random.sample(population=range(0, int(self.query_len)), k=excess)
        self.bps = np.delete(self.bps, drop_indexes, axis=None)
