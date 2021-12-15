'''
Created on 13 Dec 2021

@author: aftab
'''
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import neo4jPy as conn
import DataPreprocessesing as dataPrep



class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 30)
        self.conv2 = GCNConv(30, 10)
        self.conv3 = GCNConv(10, dataset.num_classes)

    def forward(self, data):
        #print(data)
        x, edge_index = data.x, data.edge_index
        #print('Node Matrix:',x)
        #print('Edge Matrix in COO format:',edge_index)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)

getData = conn.HelloWorldExample("bolt://localhost:7687", "neo4j", "123456")
neodata = getData.print_greeting()
#print(neodata)
getData.close()
preprocess=dataPrep.DataPrep()
dataset=preprocess.convertData(neodata)
print(dataset)
if dataset:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)
   
    data = dataset.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')