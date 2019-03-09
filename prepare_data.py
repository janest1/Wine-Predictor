""""""

import csv
import spacy
import numpy as np


def get_input_vectors(dataset):
    """Get matrix of dense representations of each instance in dataset to use as input to neural classifier"""

    # TODO add other features besides raw text?
    # TODO token embeddings (concatenated?) instead of full wine description

    print('...setting up input vectors...')

    embeddings = spacy.load('en_core_web_md')
    vectors = np.zeros((len(dataset), 300))

    for i in range(len(dataset)):
        instance = dataset[i]
        # add vector for current instance to overall matrix
        vectors[i] = embeddings(instance['description']).vector

    return vectors


def get_label_vectors(dataset, labels):
    """Get sparse representations of the label (wine variety) for each instance in dataset"""

    label_vectors = []
    for instance in dataset:
        label = np.zeros(len(labels))
        variety = instance['variety_normalized']
        index = labels.index(variety)
        label[index] = 1
        label_vectors.append(label)

    return np.array(label_vectors)


def get_labels(dataset):
    """"""

    labels = []
    for instance in dataset:
        if instance['variety_normalized'] not in labels:
            labels.append(instance['variety_normalized'])

    return labels


if __name__ == '__main__':

    # with open('winemag-data_first150k.csv') as infile:
    with open('wine-reviews/wines_updated.csv') as infile:
        reader = csv.DictReader(infile)
        total_data = list(reader)
        # total_data = [dict(line) for line in reader]

    # split data into train, dev, and test
    train_size = round(0.8 * len(total_data))
    train = total_data[:train_size]
    remaining = total_data[train_size:]
    dev_size = round(len(remaining) / 2)
    dev = remaining[:dev_size]
    test = remaining[dev_size:]

    # get vector representations of each instance in each dataset; save to files
    train_vectors = get_input_vectors(train)
    np.save('train_vectors.npy', train_vectors)

    dev_vectors = get_input_vectors(dev)
    np.save('dev_vectors.npy', dev_vectors)

    test_vectors = get_input_vectors(test)
    np.save('test_vectors.npy', test_vectors)

    # get vector representation of correct label for each data instance; save to file
    labels = get_labels(train) + get_labels(dev) + get_labels(test)
    train_label_vectors = get_label_vectors(train, labels)
    np.save('train_labels.npy', train_label_vectors)
    dev_label_vectors = get_label_vectors(dev, labels)
    np.save('dev_labels.npy', dev_label_vectors)
    test_label_vectors = get_label_vectors(test, labels)
    np.save('test_labels.npy', test_label_vectors)
