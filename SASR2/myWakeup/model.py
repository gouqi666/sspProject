import  torch
from torch import  nn
from torch.nn.utils import weight_norm
# class Wake_UP_Model(nn.Module):
#     def __init__(self):

class DNN(nn.Module):
    def __init__(self,input_size,hidden_size,target_size = 2):
        super(DNN,self).__init__()
        self.hidden1 = nn.Linear(input_size,hidden_size)
        self.hidden2 = nn.Linear(hidden_size,10000)
        self.hidden3 = nn.Linear(10000,5000)
        self.hidden4 = nn.Linear(5000,500)
        self.hidden5 = nn.Linear(500,50)
        self.hidden6 = nn.Linear(50,10)
        self.hidden7 = nn.Linear(10,target_size)
        self.act = nn.ReLU()
    def forward(self,x):
        out = self.act(self.hidden1(x))
        out = self.act(self.hidden2(out))
        out = self.act(self.hidden3(out))
        out = self.act(self.hidden4(out))
        out = self.act(self.hidden5(out))
        out = self.act(self.hidden6(out))
        return self.hidden7(out)





# 定义一个卷积块，指定激活函数为GLU
class ConvBlock(nn.Module):
    def __init__(self, conv, p):
        super().__init__()
        self.conv = conv
        nn.init.kaiming_normal_(self.conv.weight)
        self.conv = weight_norm(self.conv)
        self.act = nn.GLU(1)
        self.dropout = nn.Dropout(p, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class GatedConv(nn.Module):
    """ 这是一个介于Wav2letter和门控Convnets之间的模型。该模型的核心模块是门控卷积网络 """

    def __init__(self):
        """ vocabulary : str : string of all labels such that vocaulary[0] == ctc_blank  """
        super(GatedConv,self).__init__()
        output_units = 2
        # 创建卷积网络
        modules = [ConvBlock(nn.Conv1d(161, 500, 48, 2, 97), 0.2)]
        for i in range(7):
            modules.append(ConvBlock(nn.Conv1d(250, 500, 7, 1), 0.3))
        modules.append(ConvBlock(nn.Conv1d(250, 2000, 32, 1), 0.5))
        modules.append(ConvBlock(nn.Conv1d(1000, 2000, 1, 1), 0.5))
        modules.append(weight_norm(nn.Conv1d(1000, output_units, 1, 1)))
        # modules.append(nn.functional.adaptive_avg_pool2d(a, (1,1)))
        self.cnn = nn.Sequential(*modules)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(590,30), # 3s
            nn.ReLU(),
            nn.Linear(30,10),
            nn.ReLU(),
            nn.Linear(10,2)
        )


    def forward(self, x):  # -> B * V * T
        x = self.cnn(x)
        x = self.flatten(x)
        x= self.classifier(x)
        # for module in self.modules():
        #     if type(module) == nn.modules.Conv1d:
        #         lens = (lens - module.kernel_size[0] + 2 * module.padding[0]) // module.stride[0] + 1
        return x

    # 预测音频
    # def predict(self, path):
    #     self.eval()
    #     # 加载音频数据集并执行短时傅里叶变换
    #     wav = data.load_audio(path)
    #     spec = data.spectrogram(wav)
    #     spec.unsqueeze_(0)
    #     out = self.cnn(spec)
    #     out_len = torch.tensor([out.size(-1)])
    #     # 把预测结果转换成文字
    #     text = self.decode(out, out_len)
    #     self.train()
    #     return text[0]

