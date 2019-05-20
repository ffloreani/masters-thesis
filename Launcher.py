from tensorflow.python.keras.callbacks import ModelCheckpoint

from PAFParser import parsepaf, file_line_count
from Visualizer import visualize_json
from classifier.Model import create_model


def parse(file):
    print("Line count: {}\n".format(file_line_count(file)))
    parsepaf(file)


def visualize(file):
    visualize_json(file, start=0, end=100)


if __name__ == "__main__":
    # parse("./data/1m_sample.paf")

    visualize("./output/output_2019-05-14T20:31:45.394497.json")

    # with open("./data/output_2019-04-16T22:01:50.563342.json", "r") as f1:
    #     last_line = f1.readlines()[-1]
    #     print(last_line[-1350:])
    # model = create_model()
    #
    # callback = ModelCheckpoint(
    #     filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
    #     monitor='val_loss',
    #     save_best_only=True)
    #
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
