import json

import numpy as np
import matplotlib.pyplot as plt


def visualize_json(json_file, end, start=0):
    with open(json_file) as file_handle:
        decoded = json.load(file_handle)

        for i in range(start, end):
            query_name = decoded['sequences'][i]['id'].replace("/", "__")
            overlaps = decoded['sequences'][i]['overlaps']

            ys = np.fromstring(overlaps, sep=",")
            xs = list(range(len(ys)))

            plt.plot(xs, ys)
            plt.savefig('./charts/{}.png'.format(query_name), bbox_inches='tight')

            plt.clf()
