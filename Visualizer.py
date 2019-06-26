import csv

import matplotlib.pyplot as plt
import numpy as np


def visualize_json(json_file):
    with open(json_file) as file_handle:
        decoded = csv.reader(file_handle, delimiter="\t")

        for i, sequence in enumerate(decoded):
            query_name = sequence[0].replace("/", "__")
            overlaps = sequence[3]

            ys = np.fromstring(overlaps, sep=",")
            print("Sequence {}, size = {}".format(i, ys.size))
            xs = list(range(len(ys)))

            plt.plot(xs, ys)
            plt.savefig('./charts/{}.png'.format(query_name), bbox_inches='tight')

            plt.clf()
