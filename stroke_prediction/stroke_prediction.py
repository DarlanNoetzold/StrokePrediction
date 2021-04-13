import pickle


class StrokePrediction(object):
    def __init__(self):
        self.avg_glucose_level = pickle.load(open('parameter/avg_glucose_level.pkl', 'rb'))
        self.bmi = pickle.load(open('parameter/bmi.pkl', 'rb'))

    def data_preparation(self, df):
        df['free sulfur dioxide'] = self.avg_glucose_level.transform(df[['avg_glucose_level']].values)

        df['total sulfur dioxide'] = self.bmi.transform(df[['bmi']].values)

        df1 = df[["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]]

        return df1
