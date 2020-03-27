from sklearn.pipeline import Pipeline

class valor:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def fit(self, X, y=None):
        print(X)
        print(self.x)
        print(self.y)
        return self
    
    def transform(self, X=None):
        self.x *= 2
        self.y *= 3
        return self
        

pipe = Pipeline([
                    ('chave', valor(12,20)),
                    ('chave2', valor(120,780))
                ])

pipe.named_steps.chave.transform().fit('qwe')