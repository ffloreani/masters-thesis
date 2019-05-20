import random

import numpy as np

from classifier.Model import INPUT_LENGTH

REGULAR = "regular"
CHIMERIC = "chimeric"
REPEAT = "repeat"


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

    def print(self, file_handle, last_sequence=False):
        overlaps = ",".join(map(str, self.bps.tolist()))

        json = "{\"id\":\"%s\",\"origLen\":\"%s\",\"overlaps\":\"%s\"}" % (self.query_id, self.query_len, overlaps)

        if not last_sequence:
            json = json + ",\n"

        file_handle.write(json)

    def append(self, query_hit_start, query_hit_end):
        for index in range(query_hit_start, query_hit_end):
            self.bps[index] += 1

    def pad_sequence(self):
        padding = INPUT_LENGTH - int(self.query_len)
        if padding == 0:
            return

        # print("Padding sequence by data interpolation".format(padding * 2))
        self.bps = np.concatenate((np.zeros(padding), self.bps))
        self.bps = np.append(self.bps, np.zeros(padding))

    def trim_sequence(self):
        excess = int(self.query_len) - INPUT_LENGTH
        if excess == 0:
            return

        # print("Trimming sequnece by removing {} entries".format(excess))
        drop_indexes = random.sample(population=range(0, int(self.query_len)), k=excess)
        self.bps = np.delete(self.bps, drop_indexes, axis=None)
