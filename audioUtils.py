"""
Utility functions for audio files
"""
import librosa
import os
from tqdm import tqdm
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
from PIL import Image


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,
                 i,
                 format(cm[i, j], fmt),
                 size=11,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)
    plt.savefig('picConfMatrix.png', dpi=400)
    plt.tight_layout()


def WAV2Numpy(folder, sr=None):
    """
    Recursively converts WAV to numpy arrays.
    Deletes the WAV files in the process

    folder - folder to convert.
    """
    allFiles = []
    for root, dirs, files in os.walk(folder):
        allFiles += [os.path.join(root, f) for f in files if f.endswith('.wav')]

    for file in tqdm(allFiles):
        y, sr = librosa.load(file, sr=None)

        # resize length of 'y' to factors of 'sr'
        pad_size = int(math.ceil(len(y) / float(sr)) * sr) - len(y)
        y_pad = np.pad(y, (0, pad_size), 'constant', constant_values=(0.))

        # if we want to write the file later
        # librosa.output.write_wav('file.wav', y, sr, norm=False)
        np.save(file + '.npy', y)

        # save to gif
        # NOTE: clipped 32768 to -32768, -32768 to -32767
        a_max = (1 << ((8 * 2) - 1))
        y_int16 = (y_pad * float(a_max)).astype(np.int16)
        y_uint16 = y_int16.view(np.uint16)
        #np.info(y_uint16)
        # save bitmaps of waveform to RGX png file
        shape_rows = sr
        y_uint8_high = (y_uint16 >> 8).reshape(
            (shape_rows, -1)).astype(np.uint8)
        y_uint8_low = (y_uint16 & ((1 << (8 + 1)) - 1)).reshape(
            (shape_rows, -1)).astype(np.uint8)
        y_bitmap = np.dstack(
            (y_uint8_high, y_uint8_low, np.zeros_like(y_uint8_high)))
        #np.info(y_uint8_high)
        #np.info(y_uint8_low)
        #np.info(y_bitmap)
        im = Image.fromarray(y_bitmap, mode='RGB')
        im.save(file + '.bmp')
        np.savetxt(file + '.csv', y_int16,  delimiter=',')

        os.remove(file)
