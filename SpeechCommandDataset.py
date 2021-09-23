import os
import math
from typing import Tuple

import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F

import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

torchaudio.set_audio_backend('sox_io')

HASH_DIVIDER = "_nohash_"

GSCmdV2Categs = {
    'unknown': 0,
    'silence': 0,
    '_unknown_': 0,
    '_silence_': 0,
    '_background_noise_': 0,
    'yes': 2,
    'no': 3,
    'up': 4,
    'down': 5,
    'left': 6,
    'right': 7,
    'on': 8,
    'off': 9,
    'stop': 10,
    'go': 11,
    'zero': 12,
    'one': 13,
    'two': 14,
    'three': 15,
    'four': 16,
    'five': 17,
    'six': 18,
    'seven': 19,
    'eight': 20,
    'nine': 1
}
numGSCmdV2Categs = 21


def load_speechcommands_item(
        filepath: str,
        path: str,
        catDict: dict,
        mem_fault: str,
        cache: bool = True,
        data_quantize_bits: int = 4) -> Tuple[Tensor, int, str, str, int]:
    relpath = os.path.relpath(filepath, path)
    label, filename = os.path.split(relpath)
    # Besides the officially supported split method for datasets defined by
    # "validation_list.txt" and "testing_list.txt" over
    # "speech_commands_v0.0x.tar.gz" archives, an alternative split method
    # referred to in paragraph 2-3 of Section 7.1, references 13 and 14 of
    # the original paper, and the checksums file from the tensorflow_datasets
    # package [1] is also supported.
    # Some filenames in those "speech_commands_test_set_v0.0x.tar.gz" archives
    # have the form "xxx.wav.wav", so file extensions twice needs to be
    # stripped twice.
    # [1] https://github.com/tensorflow/datasets/blob/master
    # /tensorflow_datasets/url_checksums/speech_commands.txt
    speaker, _ = os.path.splitext(filename)
    speaker, _ = os.path.splitext(speaker)

    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    csv_filepath = filepath + '.csv'
    bins_npy_filepath = csv_filepath + '.bins.npy'
    # generate faulty waveform and repaired waveform
    faulty_csv_filepath = filepath + '_fault.csv'
    repaired_n_csv_filepath = filepath + '_repaired_n.csv'
    repaired_s_csv_filepath = filepath + '_repaired_s.csv'

    # max magnatude of int16 (32768)
    a_max = (1 << ((8 * 2) - 1))

    metadata = torchaudio.info(filepath)
    sample_rate = metadata.sample_rate

    if not os.path.isfile(bins_npy_filepath) or not cache:
        if not os.path.isfile(csv_filepath):
            # Load audio with normalization
            waveform, sample_rate = torchaudio.load(filepath)
            # resize length of 'waveform' to factors of 'sample_rate'
            pad_size = int(
                math.ceil(metadata.num_frames / float(sample_rate)) *
                sample_rate) - metadata.num_frames
            waveform_pad = F.pad(waveform, (0, pad_size))
            if __debug__ and metadata.num_frames > sample_rate:
                print(sample_rate, metadata.num_frames, waveform_pad.shape)
            # quantize to int16
            # NOTE: clipped 32768 to -32768, -32768 to -32767
            waveform_int16 = (waveform_pad * float(a_max)).to(torch.int16)
            # save to csv
            np.savetxt(csv_filepath,
                       waveform_int16.numpy(),
                       fmt='%d',
                       delimiter=',')
            # generate faulty waveform (MSB: 15th bit; LSB: 0th bit; faulty bit: 13)
            faulty_mask = np.full_like(waveform_int16, 1 << 13)
            waveform_int16_faulty = np.bitwise_or(waveform_int16, faulty_mask)
            np.savetxt(faulty_csv_filepath,
                       waveform_int16_faulty.numpy(),
                       fmt='%d',
                       delimiter=',')
            # generate repaired waveform (repair the faulty bit with its neighbor bit: 12)
            neighbor_mask = np.full_like(waveform_int16, 1 << 12)
            waveform_int16_neighbor = np.bitwise_and(waveform_int16,
                                                     neighbor_mask)
            waveform_int16_neighbor = np.left_shift(waveform_int16_neighbor, 1)
            repaired_mask = np.invert(faulty_mask)
            waveform_int16_repaired = np.bitwise_and(waveform_int16,
                                                     repaired_mask)
            waveform_int16_repaired_n = np.bitwise_or(waveform_int16_repaired,
                                                      waveform_int16_neighbor)
            np.savetxt(repaired_n_csv_filepath,
                       waveform_int16_repaired_n.numpy(),
                       fmt='%d',
                       delimiter=',')
            # generate repaired waveform (repair the faulty bit with its sign bit: 15)
            sign_mask = np.full_like(waveform_int16, 1 << 15)
            waveform_int16_sign = np.bitwise_and(waveform_int16, sign_mask)
            waveform_int16_sign = np.right_shift(waveform_int16_sign, 2)
            waveform_int16_repaired_s = np.bitwise_or(waveform_int16_repaired,
                                                      waveform_int16_sign)
            np.savetxt(repaired_s_csv_filepath,
                       waveform_int16_repaired_s.numpy(),
                       fmt='%d',
                       delimiter=',')
        else:
            waveform_int16 = torch.from_numpy(
                np.loadtxt(csv_filepath, dtype=np.int16, delimiter=','))
            waveform_int16_faulty = torch.from_numpy(
                np.loadtxt(faulty_csv_filepath, dtype=np.int16, delimiter=','))
            waveform_int16_repaired_n = torch.from_numpy(
                np.loadtxt(repaired_n_csv_filepath,
                           dtype=np.int16,
                           delimiter=','))
            waveform_int16_repaired_s = torch.from_numpy(
                np.loadtxt(repaired_s_csv_filepath,
                           dtype=np.int16,
                           delimiter=','))

        if mem_fault == "faulty":
            waveform_int16 = waveform_int16_faulty
        elif mem_fault == "reparied_n":
            waveform_int16 = waveform_int16_repaired_n
        elif mem_fault == "reparied_s":
            waveform_int16 = waveform_int16_repaired_s

        if data_quantize_bits > 0:
            # save bins map of waveform to npy file
            waveform_uint16 = torch.clamp(waveform_int16.to(torch.int32) +
                                          a_max,
                                          min=0,
                                          max=(a_max * 2 - 1)).to(torch.int64)
            waveform_bins = (waveform_uint16 >> (16 - data_quantize_bits))
            waveform_binsmap = F.one_hot(
                waveform_bins,
                num_classes=(1 << data_quantize_bits)).squeeze()
            if __debug__:
                print(sample_rate, metadata.num_frames, waveform_binsmap.shape,
                      label)
            waveform_binsmap = torch.transpose(
                waveform_binsmap, 1, 0).unsqueeze(-1).to(torch.float32)
            if cache:
                np.save(bins_npy_filepath, waveform_binsmap.numpy())
        else:
            # Load audio with normalization
            # transform buffered csv to normalization waveform
            waveform_binsmap = torch.unsqueeze(waveform_int16, 0).unsqueeze(-1)
    else:
        waveform_binsmap = torch.from_numpy(np.load(bins_npy_filepath))
    return waveform_binsmap, sample_rate, catDict.get(
        label, 0), speaker_id, utterance_number


class SpeechCommandDataset(SPEECHCOMMANDS):
    def __init__(self,
                 subset: str = None,
                 task: str = None,
                 mem_fault: str = "baseline",
                 cache: bool = True,
                 data_quantize_bits: int = 4):

        if not os.path.exists("sd_GSCmdV2/"):
            os.makedirs("sd_GSCmdV2/")

        super().__init__("./sd_GSCmdV2", download=True)

        if task == '12cmd':
            GSCmdV2Categs = {
                'unknown': 0,
                'silence': 1,
                '_unknown_': 0,
                '_silence_': 1,
                '_background_noise_': 1,
                'yes': 2,
                'no': 3,
                'up': 4,
                'down': 5,
                'left': 6,
                'right': 7,
                'on': 8,
                'off': 9,
                'stop': 10,
                'go': 11
            }
            numGSCmdV2Categs = 12
        elif task == 'leftright':
            GSCmdV2Categs = {
                'unknown': 0,
                'silence': 0,
                '_unknown_': 0,
                '_silence_': 0,
                '_background_noise_': 0,
                'left': 1,
                'right': 2
            }
            numGSCmdV2Categs = 3
        elif task == '35word':
            GSCmdV2Categs = {
                'unknown': 0,
                'silence': 0,
                '_unknown_': 0,
                '_silence_': 0,
                '_background_noise_': 0,
                'yes': 2,
                'no': 3,
                'up': 4,
                'down': 5,
                'left': 6,
                'right': 7,
                'on': 8,
                'off': 9,
                'stop': 10,
                'go': 11,
                'zero': 12,
                'one': 13,
                'two': 14,
                'three': 15,
                'four': 16,
                'five': 17,
                'six': 18,
                'seven': 19,
                'eight': 20,
                'nine': 1,
                'backward': 21,
                'bed': 22,
                'bird': 23,
                'cat': 24,
                'dog': 25,
                'follow': 26,
                'forward': 27,
                'happy': 28,
                'house': 29,
                'learn': 30,
                'marvin': 31,
                'sheila': 32,
                'tree': 33,
                'visual': 34,
                'wow': 35
            }
            numGSCmdV2Categs = 36
        elif task == '20cmd':
            GSCmdV2Categs = {
                'unknown': 0,
                'silence': 0,
                '_unknown_': 0,
                '_silence_': 0,
                '_background_noise_': 0,
                'yes': 2,
                'no': 3,
                'up': 4,
                'down': 5,
                'left': 6,
                'right': 7,
                'on': 8,
                'off': 9,
                'stop': 10,
                'go': 11,
                'zero': 12,
                'one': 13,
                'two': 14,
                'three': 15,
                'four': 16,
                'five': 17,
                'six': 18,
                'seven': 19,
                'eight': 20,
                'nine': 1
            }
            numGSCmdV2Categs = 21

        self.data_quantize_bits = data_quantize_bits
        self.task = task
        self.num_classes = numGSCmdV2Categs
        self.catDict = GSCmdV2Categs
        self.cache = cache
        self.mem_fault = mem_fault

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [
                    os.path.join(self._path, line.strip()) for line in fileobj
                ]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list(
                "testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, label,
                            speaker_id, utterance_number)``
        """
        fileid = self._walker[n]
        return load_speechcommands_item(fileid, self._path, self.catDict,
                                        self.cache, self.mem_fault,
                                        self.data_quantize_bits)
