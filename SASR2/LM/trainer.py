import numpy as np
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import lm_token_path, am_token_path, TextFeaturizer
from .config import training_config
# from .bert import bert_model, bert_tokenizer
from .dataset import LMDataSet
from .loss import BertFeatureLoss, LMCrossEntropyLoss, LMAccuracy
from .model import Transformer

am_features = TextFeaturizer(am_token_path)
lm_features = TextFeaturizer(lm_token_path)


def collate_fn(train_data):
    train_data.sort(key=lambda data: len(data[0]), reverse=True)
    x = [torch.Tensor(item[0]) for item in train_data]
    y = [torch.Tensor(item[1]) for item in train_data]
    # bert_token = [torch.Tensor(item[2]) for item in train_data]
    x = pad_sequence(x, batch_first=True, padding_value=0).int()
    y = pad_sequence(y, batch_first=True, padding_value=0).int()
    # bert_token = pad_sequence(bert_token, batch_first=True, padding_value=0).int()
    mask = torch.eq(x, 0)
    # bert_feature = bert_model.forward(bert_token.cuda(), attention_mask=torch.logical_not(mask).int().cuda())
    train_data = (x, y)
    return train_data, mask


class LMTrainer:
    def __init__(self):
        self.amf = TextFeaturizer(am_token_path)
        self.lmf = TextFeaturizer(lm_token_path)
        self.train_data = DataLoader(LMDataSet(self.amf, self.lmf, True),
                                     batch_size=training_config["batch_size"],
                                     shuffle=True, collate_fn=collate_fn)
        self.valid_data = DataLoader(LMDataSet(self.amf, self.lmf, False),
                                     batch_size=training_config["batch_size"],
                                     shuffle=True, collate_fn=collate_fn)
        self.model = Transformer(self.amf, self.lmf)
        self.save_path = training_config["save_path"]
        self.epoch = training_config["epoch"]
        self.loaded = False

    def train(self, resume=True):
        # print(torch.cuda.current_device())
        with torch.cuda.device("cuda:0"):
            self.model = self.model.cuda()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            warm_up_ratio = 0.1
            warm_up_steps = int(self.epoch * len(self.train_data) * warm_up_ratio)
            train_steps = int(self.epoch * len(self.train_data) * (1 - warm_up_ratio))
            scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                     num_warmup_steps=warm_up_steps,
                                                                     num_training_steps=train_steps)

            bert_feature_loss = BertFeatureLoss()
            cross_entropy_loss = LMCrossEntropyLoss()
            metrics = LMAccuracy()

            if resume:  # restore from checkpoint
                self.model, optimizer, epoch = self.restore_from(self.model, optimizer, self.save_path)
                self.loaded = True

            for epoch in range(self.epoch):

                train_loss = []
                self.model.train()
                print("epoch:{}".format(epoch))
                with tqdm(total=len(self.train_data)) as bar:
                    for i, batch in enumerate(self.train_data):
                        (x, y), mask = batch
                        x, y, mask = x.cuda(), y.cuda(), mask.cuda()
                        optimizer.zero_grad()
                        output_classes, features = self.model(x, mask)
                        class_loss = cross_entropy_loss(output_classes, y, mask)
                        acc = metrics(output_classes, y, mask)
                        # feature_loss = bert_feature_loss(features, bert_feature, mask)
                        feature_loss = 0
                        loss = class_loss + feature_loss
                        loss.backward()
                        optimizer.step()
                        train_loss.append(loss.item())
                        bar.set_postfix(train_loss=loss, acc=acc)
                        bar.update(1)
                        scheduler.step()
                    train_loss = np.mean(train_loss)

                valid_loss = []
                accs = []
                self.model.eval()  # 注意model的模式从train()变成了eval()
                for i, batch in enumerate(tqdm(self.valid_data)):
                    (x, y), mask = batch
                    x, y, mask = x.cuda(), y.cuda(), mask.cuda()
                    # output_classes, features = self.model(x.cuda(), mask.cuda())
                    output_classes, features = self.model(x, mask)
                    class_loss = cross_entropy_loss(output_classes, y, mask)
                    acc = metrics(output_classes, y, mask)
                    # feature_loss = bert_feature_loss(features, bert_feature, mask)
                    feature_loss = 0
                    loss = class_loss + feature_loss
                    valid_loss.append(loss.item())
                    accs.append(acc.item())
                valid_loss = np.mean(valid_loss)
                acc = np.mean(accs)

                print("train loss: {} valid loss: {} acc: {}".format(train_loss, valid_loss, acc))

                torch.save(
                    {'epoch': epoch,
                     'state_dict': self.model.state_dict(),
                     'optimizer': optimizer.state_dict()},
                    self.save_path)
                print("模型已保存")

    def predict(self, data, load=True):
        if load and not self.loaded:
            self.model = self.restore_from(self.model)
            self.loaded = True
        if len(data) > 0 and isinstance(data[0], str):
            data = [data]
        sentence = []
        for one_data in data:
            pinyin_token = torch.tensor([am_features.encode(one_data)])
            out_classes, _ = self.model(pinyin_token)
            words = lm_features.decode(torch.argmax(out_classes, -1).numpy()[0])
            sentence.append(words)
        return sentence

    def restore_from(self, model, optimizer=None, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = self.save_path
        # device = torch.cuda.current_device()
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        epoch = ckpt['epoch']
        ckpt_model_dict = ckpt['state_dict']
        model.load_state_dict(ckpt_model_dict, strict=False)  # load model
        if optimizer:
            optimizer.load_state_dict(ckpt['optimizer'])  # load optimizer
            return model, optimizer, epoch
        else:
            return model
