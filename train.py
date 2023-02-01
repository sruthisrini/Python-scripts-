from model import RNNDataset, LSTMModel
from transform import prepare_data
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss,MultiLabelSoftMarginLoss
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import wandb

wandb.init(project="my-project")
criterion=MultiLabelSoftMarginLoss()
model_path=r"D:\multi_label_classification\trained_model.pt"
def train(model, optimizer, train_loader, criterion, device):
    model.train()
    loss_all = 0
    batch_size=1
    train_acc=0
    for batch_data, batch_labels, batch_lens in train_loader:
        optimizer.zero_grad()
        outputs, _ = model(batch_data, batch_lens, len(batch_labels))
        outputs = outputs.reshape([batch_size, 1,30])
        loss = criterion(outputs, batch_labels)
        loss_all += loss.item() * len(batch_labels)
        loss.backward()
        optimizer.step()
        wandb.log({"train loss" : loss})
        acc = binary_accuracy(outputs, batch_labels)
        train_acc += acc.item()
    train_acc = train_acc / (len(train_loader)*batch_size)
    wandb.log({"train accuracy Multilabel" : train_acc})
    return train_acc, loss_all / len(train_loader.dataset)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum()/len(correct)
    return acc

def test(test_loader, model, criterion, device,apply_tanh=False):
    model.eval()
    loss_all = 0
    score_all = []
    test_labels = []
    test_acc = 0.0
    for batch_data, batch_labels, batch_lens in test_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        outputs, _ = model(batch_data, batch_lens, len(batch_labels))
        loss = criterion(outputs, batch_labels)
        loss_all += loss.item() * len(batch_labels)
        acc = binary_accuracy(outputs, batch_labels)
        test_acc += acc.item()
        # Use tanh to normalize scores from -1 to 1.
        if apply_tanh:
            output = torch.tanh(outputs).cpu().detach().numpy()
        else:
            output = outputs.cpu().detach().numpy()
        score_all.extend(output)
        test_labels.extend(batch_labels.cpu().detach().numpy())
    print("len:",len(test_loader))
    test_acc = test_acc / (len(test_loader)*batch_size)
    wandb.log({"test accuracy multilabel" : test_acc})
    print("test acc:")
    return test_acc

def pad_collate(batch):
    (xs, ys) = zip(*batch)
    xs_lens = [len(x) for x in xs]
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = torch.FloatTensor([[y] for y in ys])
    return xs_pad, ys,xs_lens

if __name__ == "__main__":
    csv_file_path = r"D:\University_of_Freiburg\Semester-4\project_2\K562\K562_sequenced_data.csv"
    rna_vecs,rna_labels = prepare_data(csv_file_path)
    projmlc_dataset = RNNDataset(rna_vecs, rna_labels)
    projmlc_model = LSTMModel(input_dim=4, n_class=30, device="cpu")

    config = wandb.config          
    config.batch_size = 1          
    config.epochs = 10             
    config.lr = 0.001               

    batch_size=1
    train_dataset,test_dataset=train_test_split(projmlc_dataset,test_size=0.2,random_state=0)
    train_dataset,val_dataset=train_test_split(train_dataset,test_size=0.25,random_state=0)

    #batch loaders
    train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=config.batch_size,
                                  collate_fn=pad_collate, pin_memory=True)
    
    test_loader=DataLoader(dataset=test_dataset,batch_size=config.batch_size,collate_fn=pad_collate,pin_memory=True)
    val_loader=DataLoader(dataset=val_dataset,batch_size=config.batch_size,collate_fn=pad_collate,pin_memory=True)
 
    optimizer = torch.optim.AdamW(projmlc_model.parameters(), config.lr)

    model_path=r"D:\multi_label_classification\trained_model.pt"
    wandb.watch(projmlc_model, log="all")

    for i in range(1,config.epochs + 1):
        print(train(model=projmlc_model,optimizer=optimizer, train_loader=train_loader, criterion=criterion, device="cpu"))
        print(test(test_loader=val_loader,model=projmlc_model,criterion=criterion,device="cpu"))

    torch.save(projmlc_model.state_dict(), model_path)
    wandb.save('model.h5')