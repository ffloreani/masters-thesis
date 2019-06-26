import numpy as np

from PAFParser import parse_paf, file_line_count
from Visualizer import visualize_json
from classifier.Classifier import create_datasets, evaluate, evaluate_autoencoder
from classifier.Model import create_rnn_model, create_1d_conv_model, create_deep_autoencoder


def parse(file):
    print("Line count: {}\n".format(file_line_count(file)))
    parse_paf(file)


def visualize(file):
    visualize_json(file)


def test_conv1d(dataset):
    model = create_1d_conv_model()

    train_x, train_y, test_x, test_y = create_datasets(dataset)

    print(train_x.shape)
    print(np.max(train_x))
    train_x = train_x / np.max(train_x)
    test_x = test_x / np.max(test_x)
    print(np.max(train_x))

    evaluate(model, train_x, train_y, test_x, test_y, epochs_num=9)


def test_lstm(dataset):
    model = create_rnn_model()

    train_x, train_y, test_x, test_y = create_datasets(dataset)

    print(np.max(train_x))
    train_x = train_x / np.max(train_x)
    test_x = test_x / np.max(test_x)
    print(np.max(train_x))

    train_x = np.expand_dims(train_x, axis=2)
    test_x = np.expand_dims(test_x, axis=2)
    print(train_x.shape)

    evaluate(model, train_x, train_y, test_x, test_y, epochs_num=20)


def test_autoencoder(dataset):
    model = create_deep_autoencoder()

    train_x, _, test_x, _ = create_datasets(dataset)

    print(np.max(train_x))
    train_x = train_x / np.max(train_x)
    test_x = test_x / np.max(test_x)
    print(np.max(train_x))

    evaluate_autoencoder(model, train_x, train_x, test_x, test_x, epochs_num=50)


if __name__ == "__main__":
    # parse("./data/1m_sample.paf")
    # visualize("./output/classified_5000.tsv")

    test_conv1d("./output/categorized_400.tsv")
