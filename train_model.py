import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Carregando o dataset
df = pd.read_csv('heart.csv')
print(df.head())  # Verifique se o cabeçalho está sendo impresso

# Np.array transforma em numpy

X = np.array(df.loc[ :, df.columns != 'output']) #Pego todas as colunas exceto a "Heart Attack Risk"
y =  np.array(df['output'])

print(f"X: {X.shape}, y:{y.shape}")

# Dividindo o dataset em teste / treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Normalizando os dados -> Serve para igualar as grandezas para o calculo
# Só funciona com valores inteiros
scaler =  StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale =  scaler.transform(X_test)

# Classe de treinamento da rede
class NeuralNetworkFromScratch:
    #Inicializando os parametros da rede
    def __init__(self, learning_rate, X_train, y_train, X_test, y_test):
        self.weights = np.random.randn(X_train_scale.shape[1])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.loss_train = []
        self.loss_test = []

    # Função de ativação
    def activation(self, x):
        # sigmoid: https://www.delftstack.com/pt/howto/python/sigmoid-function-python/
        return 1 / (1 + np.exp(-x))
    
    def d_activation(self, x):
        # derivate of sigmoid: https://builtin-com.translate.goog/articles/derivative-of-sigmoid-function?_x_tr_sl=en&_x_tr_tl=pt&_x_tr_hl=pt&_x_tr_pto=wa
        return self.activation(x) * (1-self.activation(x))
    
    # Backpropagation: Forward pass e Backward Pass: https://napoleon.com.br/glossario/o-que-e-backpropagation/#:~:text=O%20Backpropagation%20funciona%20em%20duas,que%20se%20obtenha%20uma%20saída.
    # Camada de rede simples
    def forward(self, X):
        hidden1 = np.dot(X, self.weights) + self.bias #multiplico o valor pelo peso + o bias (ax + b -> função linear)
        activation1 = self.activation(hidden1)
        return activation1

    def backward(self, X, y_true):
        # Calculando gradientes
        hidden1 = np.dot(X, self.weights) + self.bias #multiplico o valor pelo peso + o bias (ax + b -> função linear)
        y_pred = self.forward(X)

        derivate_loss_derivate_predicted = 2 * (y_pred - y_true)
        derivate_loss_dhidden1 =  self.d_activation(hidden1)
        derivate_hidden1_db = 1
        derivate_hidden1_dw = X

        # db é a derivada em relação ao bias, ou seja, a cada erro o bias é alterado de acordo com o valor do erro
        dL_db = derivate_loss_derivate_predicted * derivate_hidden1_db * derivate_loss_dhidden1
        
        # dw é a derivada em relação aos pesos, ou seja, a cada erro os pesos são alterados de acordo com o valor do erro
        dL_dw = derivate_loss_derivate_predicted * derivate_hidden1_dw * derivate_loss_dhidden1

        return dL_db, dL_dw
    

    def optimizer(self, dL_db, dL_dw):
        #atualizo o pesos
        self.bias = self.bias - dL_db * self.learning_rate * self.learning_rate
        self.weights = self.weights - dL_dw * self.learning_rate * self.learning_rate

    def train(self, ITERATIONS):
        for i in range(ITERATIONS):
            random_position = np.random.randint(len(self.X_train)) # escolho um valor aleatório do tamanho da tabela de treino

            #forward pass
            y_train_true =  self.X_train[random_position] # pego os valores na posição da tabela com os valores verdadeiro na posição sorteada
            y_train_pred = self.forward(self.X_train[random_position]) # pego os valores preditos pela rede na posição sorteada

            #calculo o erro entre o predito e o real usando TSE
            # Erro Quadrático Total (Total Squared Error): https://en-m-wikipedia-org.translate.goog/wiki/Mean_squared_error?_x_tr_sl=en&_x_tr_tl=pt&_x_tr_hl=pt&_x_tr_pto=tc
            loss = np.sum(np.square(y_train_pred - y_train_true))
            self.loss_train.append(loss) #adiciono a lista de erros de treino

            # calculando os gradientes
            dL_db, dL_dw = self.backward(X_train[random_position], y_train[random_position])

            # atualizo os pesos
            self.optimizer(dL_db,dL_dw)

            #calculo o erro para a divisão de teste
            loss_sum = 0
            for j in range(len(self.X_test)):
                y_true = self.y_test[j]
                y_pred = self.forward(self.X_test[j])

                loss_sum += np.square(y_pred - y_true)
            self.loss_test.append(loss_sum)

            return "=== TREINAMENTO CONCLUÍDO ==="
        

# Definindo hyperparametros
learning_rate = 0.1
ITERATIONS = 1000

# Instanciando modelo e treinando
nn = NeuralNetworkFromScratch(learning_rate=learning_rate, X_train=X_train_scale, y_train=y_train, X_test=X_test_scale, y_test=y_test)
print(nn.train(ITERATIONS=ITERATIONS))

print("Última loss de treino:", nn.loss_train[-1])
print("Última loss de teste:", nn.loss_test[-1])