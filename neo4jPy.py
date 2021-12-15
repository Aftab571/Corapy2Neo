'''
Created on 13 Dec 2021

@author: aftab
'''
from neo4j import GraphDatabase
from numpy import record
from sklearn import preprocessing




class HelloWorldExample:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def print_greeting(self):
        with self.driver.session() as session:
            greeting = session.write_transaction(self._create_and_return_greeting)
            return greeting
            

    @staticmethod
    def _create_and_return_greeting(tx):
        #result = tx.run("Match (n:Patient)-[r]->(m)Return n,r,m LIMIT 25")
        
        result = tx.run("MATCH (p1:paper)-[r:cites]->(p2:paper) RETURN p1,p2,r,ID(p1) as start ,ID(p2) as end")
        
        nodeResult = tx.run("MATCH (n) RETURN n ")

        #data = [record for record in result.data()]
        #print(data)
        '''for record in result:
            print(record['p1'].get('features'))
            print(record['p1.ID'])'''
        xList=[]
        eList=[]
        lList=[]
        le = preprocessing.LabelEncoder()
        
        for record in result:
            eList.append([record['p1'].get('edgeIndex'),record['p2'].get('edgeIndex')])
            eList.append([record['p2'].get('edgeIndex'),record['p1'].get('edgeIndex')])
            
        for node in nodeResult:
            xList.append(node['n'].get('features'))
            lList.append(node['n'].get('subject'))
        le.fit(lList)
        labelList= le.transform(lList)
        return [xList,eList,labelList]
    


if __name__ == "__main__":
    greeter = HelloWorldExample("bolt://localhost:7687", "neo4j", "123456")
    greeter.print_greeting()
    greeter.close()