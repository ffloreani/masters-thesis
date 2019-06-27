import random

import matplotlib.pyplot as plt
import numpy as np


def pad_sequence(vals, query_len):
    padding = int((1000 - int(query_len)) / 2)

    temp = np.concatenate((np.zeros(padding), vals))
    temp = np.append(temp, np.zeros(padding))

    return temp


def trim_sequence(vals, query_len):
    excess = int(query_len) - 1000

    drop_indexes = random.sample(population=range(0, int(query_len)), k=excess)
    print(drop_indexes)

    return np.delete(vals, drop_indexes, axis=None)


def randomizer():
    interpolation_indices = list(random.choices(population=range(0, 10), k=20))
    print(interpolation_indices)
    print(len(interpolation_indices))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot(func, yaxis=(-1.4, 1.4)):
    plt.ylim(yaxis)
    plt.locator_params(nbins=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axhline(lw=1, c='black')
    plt.axvline(lw=1, c='black')
    plt.grid(alpha=0.4, ls='-.')
    plt.box(on=None)
    plt.plot(x, func(x), c='r', lw=3)


if __name__ == "__main__":
    x = np.arange(-5, 5, 0.01)
    randomizer()
