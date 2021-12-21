import torch
import os
import glob
from torch.utils.data import Dataset,DataLoader
from utils import load_audio,spectrogram
class WakeUpDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __getitem__(self, item):
        return [self.x[item],self.y[item]]
    def __len__(self):
        return len(self.x)



# 音频数据加载器
class MASRDataset(Dataset):
    def __init__(self,data_path):
        pos_data_path = os.path.join(data_path, 'positive')
        neg_data_path = os.path.join(data_path, 'negative')
        x, y = [], []

        pos_wav_files = sorted(glob.glob(pos_data_path + "/*.wav"))
        neg_wav_files = sorted(glob.glob(neg_data_path + "/*.wav"))
        for wav in pos_wav_files:
            wav = load_audio(wav)
            spect = spectrogram(wav)
            x.append(spect)
            y.append(1)
        for wav in neg_wav_files:
            wav = load_audio(wav)
            spect = spectrogram(wav)
            x.append(spect)
            y.append(0)
        self.x = x
        self.y = y


    def __getitem__(self, index):

        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, freq_size, max_seqlength)
    input_lens = torch.IntTensor(minibatch_size)
    target_lens = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x].narrow(1, 0, seq_length).copy_(tensor)
        targets.append(target)
    targets = torch.IntTensor(targets)
    return inputs, targets


class MASRDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(MASRDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn