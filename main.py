import requests
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#from sklearn.tree import export_graphviz
#import graphviz


#Ajustando os Dados
dados = pd.read_csv("data/healthcare-dataset-stroke-data.csv")
dados= dados.dropna()

mms = MinMaxScaler()
dados['avg_glucose_level'] = mms.fit_transform(dados[['avg_glucose_level']].values)
pickle.dump(mms, open('parameter/avg_glucose_level.pkl', 'wb'))

dados['bmi'] = mms.fit_transform(dados[['bmi']].values)
pickle.dump(mms, open('parameter/bmi.pkl', 'wb'))


x = dados[["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]]
y = dados["stroke"]


raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25,
                                                         stratify = y)

#Treinando Modelo
modelo = DecisionTreeClassifier(max_depth=3)
modelo.fit(raw_treino_x, treino_y)
previsoes = modelo.predict(raw_teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

#Exportando o Grafico
#features = x.columns
#dot_data = export_graphviz(modelo, out_file=None,
#                           filled = True, rounded = True,
#                           feature_names = features,
#                          class_names = ["não", "sim"])
#grafico = graphviz.Source(dot_data, filename="test.gv", format="png")
#grafico.view()

#Exportando o modelo
pickle.dump(modelo, open('model/model_stroke_prediction.pkl', 'wb'))

#Fazendo request
x_json = x.to_json(orient='records')

url = 'http://192.168.1.105:5000/predict'
data = x_json
header = {'Content-type': 'application/json'}

#r = requests.post(url=url, data=data, headers=header)

#print(r)