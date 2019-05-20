from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense

INPUT_LENGTH = 3000

def create_model():
    model = Sequential()

    # Input shape (3000, 1)
    model.add(
        Conv1D(filters=80, kernel_size=20, activation='relu', input_shape=(INPUT_LENGTH, 1))
    )
    model.add(
        Conv1D(filters=80, kernel_size=20, activation='relu')
    )
    model.add(
        MaxPooling1D(pool_size=3)
    )
    model.add(
        Conv1D(filters=120, kernel_size=20, activation='relu')
    )
    model.add(
        Conv1D(filters=120, kernel_size=20, activation='relu')
    )
    model.add(
        GlobalAveragePooling1D()
    )
    model.add(
        Dropout(rate=0.5)
    )
    model.add(
        Dense(units=4, activation='softmax')
    )

    print(model.summary())

    return model
