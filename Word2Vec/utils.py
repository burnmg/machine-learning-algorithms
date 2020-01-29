import urllib.request
import os
import zipfile
import tensorflow as tf
import collections


# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str_any(f.read(f.namelist()[0])).split()
    return data


def maybe_download(filename, url, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


def build_dataset(words, vocab_size, max_words_count=float("inf")):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    # n_words - 1 because UNK is also part of the words
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0

    word_count = 0
    for word in words:
        if word_count >= max_words_count:
            break
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
        word_count += 1

    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def collect_data(vocabulary_size=10000, max_words_len=float("inf")):
    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download('text8.zip', url, 31344016)
    words = read_data(filename)
    data, count, dictionary, reverse_dictionary = build_dataset(words,
                                                                vocabulary_size,
                                                                max_words_len)
    del words  # Hint to reduce memory.,
    return data, count, dictionary, reverse_dictionary

