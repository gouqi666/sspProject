from pypinyin import pinyin
from torch.utils import data

# from .bert import bert_model, bert_tokenizer
from utils import TextFeaturizer
from .config import train_path, eval_path


class LMDataSet(data.Dataset):
    def __init__(self, am_featurizer: TextFeaturizer, lm_featurizer: TextFeaturizer, train=True):
        if train:
            self.dataset = self.load_file(train_path)
        else:
            self.dataset = self.load_file(eval_path)
        self.amf = am_featurizer
        self.lmf = lm_featurizer

    @staticmethod
    def load_file(path):
        lines = open(path, 'r', encoding="utf8").readlines()
        return [line.strip().split() for line in lines]

    def __getitem__(self, item):
        y = self.dataset[item]
        y = ''.join(y)
        x = pinyin(self.dataset[item])
        x = [item[0] for item in x]
        return self.amf.encode(x), self.lmf.encode(y)

    def __len__(self):
        return len(self.dataset)


