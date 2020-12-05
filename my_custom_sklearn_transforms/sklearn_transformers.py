from sklearn.base import BaseEstimator, TransformerMixin

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        data = data.drop(labels=self.columns, axis='columns')
        data = data.replace(to_replace={'FALSE POSITIVE': 1, 'CANDIDATE': 2})
        # Devolvemos un nuevo dataframe de datos preprocesados
        return data
