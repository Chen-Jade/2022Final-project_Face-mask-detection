import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model import VGG16_model

epochs = 12


def draw_loss(train_loss, test_loss):
    plt.figure()
    plt.title('Loss')
    plt.plot(range(epochs), train_loss, label='train_loss = %0.4f' % train_loss)
    plt.plot(range(epochs), test_loss, label='test_loss = %0.4f' % test_loss)
    plt.legend()
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()


def draw_accuracy(train_accuracy, test_accuracy):
    plt.figure()
    plt.title('Accuracy')
    plt.plot(range(epochs), train_accuracy, label='train_accuracy = %0.4f' % train_accuracy)
    plt.plot(range(epochs), test_accuracy, label='test_accuracy = %0.4f' % test_accuracy)
    plt.legend()
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()


def split_dataset():
    dataset_dir = "Dataset_part"
    mask_dir = os.path.join(dataset_dir, 'with_mask')
    no_mask_dir = os.path.join(dataset_dir, 'without_mask')
    mask_incorrect_dir = os.path.join(dataset_dir, "mask_weared_incorrect")

    mask_path = [os.path.abspath(fp) for fp in glob.glob(os.path.join(mask_dir, '*.png'))]
    no_mask_path = [os.path.abspath(fp) for fp in glob.glob(os.path.join(no_mask_dir, '*.png'))]
    no_mask_path = no_mask_path + [os.path.abspath(fp) for fp in glob.glob(os.path.join(mask_incorrect_dir, '*.png'))]
    all_path = mask_path + no_mask_path

    all_label = [1] * len(mask_path) + [0] * len(no_mask_path)

    train_path, test_path, train_label, test_label = train_test_split(all_path, all_label, shuffle=True,
                                                                      train_size=0.9)
    return train_path, test_path, train_label, test_label


def preprocess_dataset(path, label):
    data_set = []
    label_set = []
    for (one_path, one_label) in zip(path, label):
        image = cv2.imread(one_path)

        image = image / 255.
        data_set.append(image)
        label_set.append(one_label)

    return np.array(data_set), np.array(label_set)


if __name__ == '__main__':
    train_path, test_path, train_label, test_label = split_dataset()
    train_data, train_label = preprocess_dataset(train_path, train_label)
    test_data, test_label = preprocess_dataset(test_path, test_label)
    print("Number of training set images：", int(train_data.shape[0] * 0.9))
    print("Number of validation set images：", int(train_data.shape[0] * 0.1))
    print("Number of test set images：", test_data.shape[0])
    model = VGG16_model
    output = model.fit(x=train_data,
                       y=train_label,
                       batch_size=32,
                       epochs=epochs,
                       verbose=1,
                       validation_split=0.1
                       )
    model.save("mask.h5")

    history_predict = output.history
    train_loss = history_predict['loss']
    train_accuracy = history_predict['accuracy']
    test_loss = history_predict['val_loss']
    test_accuracy = history_predict['val_accuracy']

    draw_loss(train_loss, test_loss)
    draw_accuracy(train_accuracy, test_accuracy)

    acc = model.evaluate(test_data, test_label, batch_size=32, verbose=1)
    print(" test accuracy:", acc[1])
