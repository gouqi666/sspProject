import os.path
import torch
from dataset import WakeUpDataset,MASRDataset,MASRDataLoader
from torch.utils.data import DataLoader
from preprocess import read_wav_files
import glob
from model import DNN,GatedConv
input_size = 7700  #
def collate_fn(batch):
    x,y = zip(*batch)
    x = list(x)
    max_len = input_size
    for i in range(len(x)):
        if len(x[i]) > max_len:
            x[i] = x[i][:max_len]
        else:
            x[i] = x[i] + [0] * (max_len - len(x[i]))
    return x,y
if __name__ == "__main__":
    data_path = "train_data"


    # 1. DNN + RELU
    # x,y = read_wav_files(data_path)
    # train_dataset = WakeUpDataset(x,y)
    # train_data_loader = DataLoader(train_dataset,shuffle=True,batch_size=16,collate_fn=collate_fn)
    # model = DNN(input_size,hidden_size=10000)
    # 2. CNN + GLU
    train_dataset = MASRDataset(data_path)
    train_dataloader = MASRDataLoader(train_dataset, batch_size=8,shuffle=True)
    model = GatedConv()
    # if os.path.exists('./model'):
    #     model.load_state_dict(torch.load('./model/best_model.pt'))
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 10
    optimizer = torch.optim.Adam(params=model.parameters(),lr=3e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,80],gamma = 0.95)
    global_step = 0
    min_loss = float('inf')
    for epoch in range(epochs):
        train_loss = []
        for x,y in train_dataloader:
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.long)
            pred  = model(x)
            loss = criterion(pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(loss)
            global_step += 1
            print("step:%d,loss:%.5f" % (global_step, loss))
        mean_loss = sum(train_loss) / len(train_loss)
        print('epoch:%d,loss:%.5f' % (epoch,mean_loss))
        if not os.path.exists('./model'):
            os.mkdir('./model')
        if mean_loss < min_loss:
            min_loss = mean_loss
            torch.save(model.state_dict(),'./model/best_model.pt')