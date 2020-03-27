import pandas as pd
import json

data = pd.read_csv('./AB InBev_intents.csv')

data = data[data['language'] == 'pt-BR']

data = data.reset_index()

intentList = []
for i, row in data.iterrows():
    intentList.insert(i, row['training_data'].split(',')) 
    
utterances = []
intents = []

class utterance():
    def __init__(self, text, intent):
        self.text = text
        self.intent = intent
        self.entities = []
        
class intent():
    def __init__(self, name):
        self.name = name
        
j = 0
for i, row in data.iterrows():
    for item in intentList[i]:
        utterances.insert(j, utterance(item, row['name']))
        j += 1
        
for i, row in data.iterrows():
    intents.insert(i, intent(row['name']))

json_string = json.dumps([ob.__dict__ for ob in utterances], ensure_ascii=False).encode('utf8')
json_string_intents = json.dumps([ob.__dict__ for ob in intents], ensure_ascii=False).encode('utf8')


with open('utterances.json', 'w') as file:
    file.write(json_string.decode())
    
with open('intents.json', 'w') as file:
    file.write(json_string_intents.decode())