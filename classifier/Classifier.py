import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import to_categorical

from classifier.Model import NUM_CLASSES
from model.Sequence import LOW_QUALITY, REGULAR, REPEAT


def create_datasets(tsv_input):
    data = pd.read_csv(tsv_input, delimiter="\t")

    # Filter out all low quality reads
    data = data.loc[data.CAT != LOW_QUALITY]

    # Convert sequence string to float array
    data.PTS = data.PTS.apply(string_to_array)

    # Convert labels from strings to ints
    ys = data.CAT.apply(category_to_int).to_numpy()
    xs = np.stack(data.PTS.array)

    encoded_ys = to_categorical(ys, num_classes=NUM_CLASSES)

    print("XS shape: {}".format(xs.shape))
    print("One-hot encoded YS shape: {}".format(encoded_ys.shape))

    train_x, test_x, train_y, test_y = train_test_split(xs, encoded_ys, test_size=0.15)

    return train_x, train_y, test_x, test_y


def string_to_array(data):
    data_string = str(data)
    split = data_string.split(',')

    return np.array([float(i) for i in split])


def category_to_int(data):
    category = str(data)
    if category == REGULAR:
        return 0
    elif category == REPEAT:
        return 1
    else:
        return 2


def evaluate(model, train_x, train_y, test_x, test_y, epochs_num):
    history = model.fit(
        x=train_x,
        y=train_y,
        batch_size=64,
        epochs=epochs_num,
        validation_data=(test_x, test_y)
    )
    plot_loss(history, epochs_num)

    pred_y = model.predict(test_x)
    matrix = metrics.confusion_matrix(test_y.argmax(axis=1), pred_y.argmax(axis=1))

    print("classification report:")
    print(metrics.classification_report(test_y.argmax(axis=1), pred_y.argmax(axis=1)))
    plot_confusion_matrix(matrix)


def evaluate_autoencoder(model, train_x, train_y, test_x, test_y, epochs_num):
    history = model.fit(
        x=train_x,
        y=train_y,
        batch_size=64,
        epochs=epochs_num,
        validation_data=(test_x, test_y)
    )

    decoded_x = model.predict(test_x)
    print(decoded_x.shape)
    print(decoded_x[0].tolist())

    plot_decoded(test_x, decoded_x)

    plot_loss(history, epochs_num)


def plot_decoded(original_x, decoded_x):
    n = 10  # how many overlaps we will display
    idxs = range(5000)
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        plt.subplot(2, n, i + 1)
        plt.plot(idxs, original_x[i].tolist())
        plt.gray()
        # display reconstruction
        plt.subplot(2, n, i + 1 + n)
        plt.plot(idxs, decoded_x[i].tolist())
        plt.gray()
    plt.show()


def plot_loss(model, epochs_num):
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    epochs = range(1, epochs_num + 1)
    plt.figure()
    plt.plot(epochs, loss, 'r-', label='Training loss')
    plt.plot(epochs, val_loss, 'b-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def plot_confusion_matrix(matrix):
    df_cm = pd.DataFrame(matrix, ["regular", "repeat", "chimeric"], ["regular", "repeat", "chimeric"])
    plt.figure(figsize=(10, 7))
    sb.heatmap(df_cm, annot=True)
    plt.show()

