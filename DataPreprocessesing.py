'''
Created on 14 Dec 2021

@author: aftab
'''
import torch
from torch_geometric.data import Data

class DataPrep:

    def convertData(self,neoData):
        '''xList=[]
        eList=[]
        for record in neoData:
            xList.append(record['p1'].get('features'))
            eList.append([record['p1.ID'],record['p2.ID']])'''
        edge_index = torch.tensor(neoData[1], dtype=torch.long)
        x = torch.tensor(neoData[0], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index.t().contiguous())
        data.num_classes=7
        data.train_mask=torch.tensor(neoData[2],dtype=torch.bool)
        data.test_mask = torch.tensor(neoData[2],dtype=torch.bool)
        print(data.train_mask.unique(return_counts=True))
        print(data.test_mask.unique(return_counts=True))
        data.y= torch.tensor(neoData[2])
        return data