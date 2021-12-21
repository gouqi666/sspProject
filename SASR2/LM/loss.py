from torch import nn
import torch


class BertFeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output_features, bert_features, masks):
        masks = torch.logical_not(masks)
        loss = torch.square(output_features - bert_features)
        loss = torch.sum(loss, -1) * masks / torch.sum(masks, -1).view(-1, 1)
        return torch.mean(torch.sum(loss, -1))


class LMCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, out_classes, target_classes, masks):
        out_classes = out_classes.view(out_classes.shape[0] * out_classes.shape[1], -1).contiguous()
        masks = torch.logical_not(masks.view(-1, 1).contiguous())
        target_classes = target_classes.view(-1).contiguous()
        return torch.sum(self.criterion(out_classes, target_classes.view(-1).long()) * masks.view(-1)) / torch.sum(masks)


class LMAccuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_classes, target_classes, masks):
        out_classes = out_classes.view(out_classes.shape[0] * out_classes.shape[1], -1).contiguous()
        out_classes = torch.argmax(out_classes, -1)
        masks = torch.logical_not(masks.view(-1, 1).contiguous())
        target_classes = target_classes.view(-1).contiguous()
        return torch.sum(torch.eq(out_classes, target_classes) * masks.view(-1)) / torch.sum(masks)