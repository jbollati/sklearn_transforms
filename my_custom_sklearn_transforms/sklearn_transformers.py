from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample

# All sklearn Transforms must have the `transform` and `fit` methods
class CSTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        
        df_majority = data[data.OBJETIVO=='Aceptado']
        df_minority = data[data.OBJETIVO=='Sospechoso']

        df_minority_upsampled = resample(df_minority, replace=True, n_samples = 8873, random_state=123)

        data = pd.concat([df_majority, df_minority_upsampled])
        data = data.drop(labels=self.columns, axis='columns')
        data = data.fillna(data.mean())
        # Devolvemos un nuevo dataframe de datos preprocesados
        return data
