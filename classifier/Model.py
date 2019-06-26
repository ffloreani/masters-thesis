from tensorflow.python.keras import Sequential, Input, Model, regularizers
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense, LSTM, Reshape, \
    BatchNormalization, UpSampling1D
from tensorflow.python.keras.optimizers import Adam

INPUT_LENGTH = 5000
INPUT_THRESHOLD = 1000

NUM_CLASSES = 3


def create_1d_conv_model():
    model = Sequential()

    model.add(
        Reshape(target_shape=(INPUT_LENGTH, 1), input_shape=(INPUT_LENGTH,))
    )
    model.add(
        Conv1D(filters=128, kernel_size=10, activation='relu', input_shape=(INPUT_LENGTH, 1), padding="same")
    )
    # model.add(
    #     MaxPooling1D(pool_size=10)
    # )
    # model.add(
    #     BatchNormalization()
    # )
    # model.add(
    #     Conv1D(filters=128, kernel_size=30, activation='relu', padding="same")
    # )
    # model.add(
    #     MaxPooling1D(pool_size=10)
    # )
    # model.add(
    #     BatchNormalization()
    # )
    model.add(
        Conv1D(filters=128, kernel_size=100, activation='relu', padding="same")
    )
    model.add(
        GlobalAveragePooling1D()
    )
    model.add(
        Dense(units=NUM_CLASSES, activation='softmax')
    )

    print(model.summary())

    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy')

    return model


def create_rnn_model():
    model = Sequential()

    model.add(
        LSTM(units=100, activation='relu', input_shape=(INPUT_LENGTH, 1))
    )
    model.add(
        Dropout(0.5)
    )
    model.add(
        Dense(100, activation='relu')
    )
    model.add(
        Dense(NUM_CLASSES, activation='softmax')
    )

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    return model


def create_deep_encoder(input_shape):
    dense1 = Dense(2500, activation="relu")(input_shape)
    dense2 = Dense(2000, activation="relu")(dense1)
    dense3 = Dense(1500, activation="relu")(dense2)
    dense4 = Dense(1000, activation="relu")(dense3)
    dense5 = Dense(500, activation="relu")(dense4)

    return dense5


def create_deep_decoder(encoder):
    dense5 = Dense(1000, activation="relu")(encoder)
    dense6 = Dense(1500, activation="relu")(dense5)
    dense7 = Dense(2000, activation="relu")(dense6)
    dense8 = Dense(2500, activation="relu")(dense7)
    output = Dense(5000, activation="sigmoid")(dense8)

    return output


def create_convolutional_encoder(input_shape):

    reshape = Reshape(target_shape=(INPUT_LENGTH, 1), input_shape=(INPUT_LENGTH,))(input_shape)
    conv1 = Conv1D(filters=32, kernel_size=30, activation='relu', input_shape=(INPUT_LENGTH, 1))(reshape)
    conv1 = MaxPooling1D(pool_size=10)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv1D(filters=64, kernel_size=30, activation='relu')(conv1)
    conv2 = MaxPooling1D(pool_size=10)(conv2)
    conv2 = BatchNormalization()(conv2)

    return conv2


def create_convolutional_decoder(encoder):
    conv1 = Conv1D(filters=32, kernel_size=42, activation='relu', input_shape=(INPUT_LENGTH, 1))(input_shape)
    conv1 = UpSampling1D(pool_size=10)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv1D(filters=64, kernel_size=59, activation='relu')(conv1)
    conv2 = MaxPooling1D(pool_size=10)(conv2)
    conv2 = BatchNormalization()(conv2)

    return conv2


def create_deep_autoencoder():
    input_shape = Input(shape=(INPUT_LENGTH, ))
    model = Model(input_shape, create_deep_decoder(create_deep_encoder(input_shape)))

    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy')

    print(model.summary())

    return model
