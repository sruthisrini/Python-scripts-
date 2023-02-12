import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from transform_test import prepare_data
from model_test import RNNDataset,LSTMModel
import wandb
from torchmetrics.classification import MultilabelAccuracy

wandb.init(project="12-feb")

criterion = BCEWithLogitsLoss()
model_path=r"D:\multi_label_classification\trained_model.pt"
def train(model, optimizer, train_loader, criterion, batch_size, device):
    model.train()
    loss_all = 0
    for batch_data, batch_labels, batch_lens in train_loader:
        optimizer.zero_grad()
        outputs, _ = model(batch_data, batch_lens, len(batch_labels))
        outputs = outputs.reshape([batch_size, 1,30])
        loss = criterion(outputs, batch_labels)
        wandb.log({"train loss" : loss})
        loss_all += loss.item() * len(batch_labels)
        loss.backward()
        optimizer.step()
#         except:
#             outputs = outputs.reshape([16, 1,30])
#             print("shape:",outputs.size())
#             loss = criterion(outputs, batch_labels)
#             print("loss:",loss)
#             loss_all += loss.item() * len(batch_labels)
#             loss.backward()
#             optimizer.step()
#             print(loss_all)
    torch.save(model.state_dict(), model_path)
    return loss_all / len(train_loader.dataset)

def test(test_loader, model, batch_size, criterion, device,
         apply_tanh=False):
    
    metric = MultilabelAccuracy(num_labels=30)
    model.eval()
    predicted_labels=[]
    true_labels=[]
    test_acc = 0.0

    for batch_data, batch_labels, batch_lens in test_loader:
        outputs, _ = model(batch_data, batch_lens, len(batch_labels))
        batch_labels = batch_labels.reshape([len(batch_lens), 30])
        acc = metric(outputs[0], batch_labels)
        test_acc += acc.item()
        #outputs=outputs.detach().numpy()
        #batch_labels=batch_labels.detach().numpy()
        #predicted_labels.append(outputs)
        #true_labels.append(batch_labels)
        #print(outputs)
        #print(batch_labels)
    
    test_acc = test_acc / len(test_loader)

    return test_acc
        
    """ count=0
    accuracy=0

    metric = MultilabelAccuracy(num_labels=30)
    for i in range(len(predicted_labels)):
        pred_batch=predicted_labels[i]
        true_batch=true_labels[i]

        print(pred_batch)
        print(true_batch)
        for j in range(len(list(pred_batch))):
            for k in range(len(list(pred_batch[j]))):
                accuracy=accuracy+float(metric(torch.tensor([pred_batch[j][k]]), torch.tensor([true_batch[k][0]])))
                #wandb.log({"accuracy":accuracy})
                count+=1
    #wandb.log({"accuracy/count" : accuracy/count})
    return accuracy/count   """  
    

def pad_collate(batch):
    (xs, ys) = zip(*batch)
    xs_lens = [len(x) for x in xs]
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = torch.FloatTensor([[y] for y in ys])
    return xs_pad, ys,xs_lens

if __name__ == "__main__":
    csv_file_path = r"D:\University_of_Freiburg\Semester-4\project_2\K562\K562_formatted_data.csv"
    rna_vecs,rna_labels = prepare_data(csv_file_path)
    projmlc_dataset = RNNDataset(rna_vecs, rna_labels)
    projmlc_model = LSTMModel(input_dim=4, n_class=30, device="cpu")
    batch_size=32

    train_dataset,test_dataset=train_test_split(projmlc_dataset,test_size=0.2,random_state=0)
    train_dataset_final, val_dataset=train_test_split(train_dataset, test_size=0.25, random_state=0)
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,collate_fn=pad_collate, pin_memory=True)
    test_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,collate_fn=pad_collate,pin_memory=True)
    optimizer = torch.optim.AdamW(projmlc_model.parameters(), lr=0.0001)
    

    for i in range(1,100):
        print("train loss",train(model=projmlc_model,optimizer=optimizer, train_loader=train_loader, criterion=criterion, batch_size=batch_size, device="cpu"))
        print("validation accuracy",test(test_loader=test_loader,model=projmlc_model,batch_size=batch_size,criterion=criterion,device="cpu"))
    #print(test(test_loader=test_loader,model=projmlc_model,criterion=criterion,device="cpu"))
    torch.save(projmlc_model.state_dict(), model_path)
    