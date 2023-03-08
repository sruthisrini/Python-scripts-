import torch
from torch import Tensor
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from transform_test import prepare_data,string_vectorizer,read_data_to_list
from model_test import RNNDataset,LSTMModel
import wandb
from torchmetrics.classification import MultilabelAccuracy
from sklearn.metrics import f1_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score

wandb.init(project="today")

def warn(*args, **kwargs):
    pass

warnings.warn = warn

criterion = BCEWithLogitsLoss() 
model_path=r"D:\multi_label_classification\march_4_with_0_grad.pt"
def train(model, optimizer, train_loader, criterion, batch_size, device):
    model.train()
    global loss_train
    loss_all_train = 0
    f1_acc=0
    
    for batch_data, batch_labels, batch_lens in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(batch_data, batch_lens, len(batch_labels))
            try:
                outputs = outputs.reshape([batch_size, 1,30])
                loss = criterion(outputs, batch_labels)
                loss_all_train += loss.item() * len(batch_labels)
                loss.backward()
                optimizer.step()
                sigmoid_outputs=torch.sigmoid(outputs)
                sigmoid_outputs[np.where(sigmoid_outputs>=0.5)]=1
                sigmoid_outputs[np.where(sigmoid_outputs<0.5)]=0
                sigmoid_outputs=sigmoid_outputs.reshape([32,30])
                batch_labels=batch_labels.reshape([32,30])
                f1_acc=f1_score(sigmoid_outputs.detach().numpy().astype(int), batch_labels.detach().numpy().astype(int), average='weighted')
                f1_acc+=f1_acc

            except:
                outputs = outputs.reshape([2, 1,30])
                loss = criterion(outputs, batch_labels)
                loss_all_train += loss.item() * len(batch_labels)
                loss.backward()
                optimizer.step()
                sigmoid_outputs=torch.sigmoid(outputs)
                sigmoid_outputs[np.where(sigmoid_outputs>=0.5)]=1
                sigmoid_outputs[np.where(sigmoid_outputs<0.5)]=0
                sigmoid_outputs=sigmoid_outputs.reshape([2,30])
                batch_labels=batch_labels.reshape([2,30])
                f1_acc=f1_score(sigmoid_outputs.detach().numpy().astype(int), batch_labels.detach().numpy().astype(int), average='weighted')
                f1_acc+=f1_acc
    torch.save(model.state_dict(), model_path)
    f1_total_accuracy=f1_acc/len(test_loader)
    loss_train=loss_all_train / len(train_loader.dataset)
    return loss_train,f1_total_accuracy

def validation(test_loader, model, batch_size, criterion, device):
    model.eval()
    loss_all_validation=0
    global loss_validation
    f1_acc=0


    for batch_data, batch_labels, batch_lens in test_loader:
        outputs, _ = model(batch_data, batch_lens, len(batch_labels))
        try:

            outputs = outputs.reshape([batch_size, 1,30])
            loss = criterion(outputs, batch_labels)
            loss_all_validation += loss.item() * len(batch_labels)
            loss.backward()
            optimizer.step()
            sigmoid_outputs=torch.sigmoid(outputs)
            sigmoid_outputs[np.where(sigmoid_outputs>=0.5)]=1
            sigmoid_outputs[np.where(sigmoid_outputs<0.5)]=0
            sigmoid_outputs=sigmoid_outputs.reshape([32,30])
            batch_labels=batch_labels.reshape([32,30])
            f1_acc=f1_score(sigmoid_outputs.detach().numpy().astype(int), batch_labels.detach().numpy().astype(int), average='weighted')
            f1_acc+=f1_acc

        except:
            outputs = outputs.reshape([17, 1,30])
            loss = criterion(outputs, batch_labels)
            loss_all_validation += loss.item() * len(batch_labels)
            loss.backward()
            optimizer.step()
            sigmoid_outputs=torch.sigmoid(outputs)
            sigmoid_outputs[np.where(sigmoid_outputs>=0.5)]=1
            sigmoid_outputs[np.where(sigmoid_outputs<0.5)]=0
            sigmoid_outputs=sigmoid_outputs.reshape([17,30])
            batch_labels=batch_labels.reshape([17,30])
            f1_acc=f1_score(sigmoid_outputs.detach().numpy().astype(int), batch_labels.detach().numpy().astype(int), average='weighted')
            f1_acc+=f1_acc

    f1_total_accuracy=f1_acc/len(test_loader)
    loss_validation=loss_all_validation / len(test_loader.dataset)
    return loss_validation,f1_total_accuracy

def binary_accuracy(preds, y):
    """
    Accuracy calculation:
    Round scores > 0 to 1, and scores <= 0 to 0 (using sigmoid function).

    """

    rounded_preds = torch.round(torch.sigmoid(preds))
    
    correct = (rounded_preds[:][0] == y[:][0]).float()
    print(y[:][0])
    acc = correct.sum()/len(correct)
    return acc

def test(test_loader, model, criterion, device):
    
    model.eval()
    test_acc = 0.0

    for batch_data, batch_labels, batch_lens in test_loader:
        outputs, _ = model(batch_data, batch_lens, len(batch_labels))
        try:
            outputs = outputs.reshape([batch_size, 30])
            batch_labels=batch_labels.reshape([batch_size, 30])
            acc = binary_accuracy(outputs, batch_labels)
            test_acc += acc.item()
          
        except:
            outputs = outputs.reshape([13, 30])
            batch_labels=batch_labels.reshape([13, 30])
            acc = binary_accuracy(outputs, batch_labels)
            test_acc += acc.item()
    
    test_acc = test_acc / len(test_loader)
    return test_acc


def pad_collate(batch):
    (xs, ys) = zip(*batch)
    xs_lens = [len(x) for x in xs]
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = torch.FloatTensor([[y] for y in ys])
    return xs_pad, ys,xs_lens

if __name__ == "__main__":
    csv_file_path = r"D:\University_of_Freiburg\Semester-4\project_2\K562\K562_formatted_data_march_1 - Copy.csv"
    rna_vecs,rna_labels = prepare_data(csv_file_path)
    projmlc_dataset = RNNDataset(rna_vecs, rna_labels)
    projmlc_model = LSTMModel(input_dim=4, n_class=30, activation='sigmoid',device="cpu")
    batch_size=32
    
    train_dataset,test_dataset=train_test_split(projmlc_dataset,test_size=0.2,random_state=0)
    train_dataset_final, val_dataset=train_test_split(train_dataset, test_size=0.2, random_state=0)  

    
 
    train_loader = DataLoader(dataset=train_dataset_final,batch_size=batch_size,collate_fn=pad_collate, pin_memory=True) 
    
    validation_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,collate_fn=pad_collate,pin_memory=True)
    
    test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,collate_fn=pad_collate,pin_memory=True)

    optimizer = torch.optim.AdamW(projmlc_model.parameters(), lr=0.00001)  

   
    for i in range(1,50):
        print("epoch",i,",","train loss",train(model=projmlc_model,optimizer=optimizer, train_loader=train_loader, criterion=criterion, batch_size=batch_size, device="cpu"))
        print("validation loss",validation(test_loader=validation_loader,model=projmlc_model,batch_size=batch_size,criterion=criterion,device="cpu"))
        wandb.log({"train loss":loss_train,"validation loss" : loss_validation})
        
    

    torch.save(projmlc_model.state_dict(), model_path)

    projmlc_model.load_state_dict(torch.load(model_path))
    projmlc_model.eval()
    print(test(test_loader=test_loader,model=projmlc_model,criterion=criterion,device="cpu"))
    

    def pred(seq):
        pred_input=string_vectorizer(seq)
        pred_input=pred_input.reshape([1,len(pred_input),4])
        outputs, _ = projmlc_model(pred_input, [len(pred_input)], 1)
        outputs=torch.sigmoid(outputs)
        print(outputs[0])
        p=outputs[0][0]
        p= [1 if i>=0.52 else 0 for i in p]
        return p
    
    df=pd.read_csv(csv_file_path)
    protein_names=df.columns.to_list()[1:]
    
    pred_label=pred("CGAAGGACATAGGCGTCATCACAATGCAATAAAGACACACACAACCACACAGACGACTCGAATGACACAGACGTCATCACCATGCAACACACAGGACACACACAACCACGCAGACGACTCGAAGGACACAGGCGTCATCACAATGCAATACACAAGACACACACAACCACGCAG")
    print([names for names, label in zip(protein_names,pred_label) if label == 1])


