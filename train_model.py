import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter

# Carregando o dataset
df = pd.read_csv('heart.csv')
print(df.head())  # Mostra as primeiras linhas do dataset

# Separando X e y
X = np.array(df.loc[:, df.columns != 'output'])  # Todos os dados exceto a coluna 'output'
y = np.array(df['output'])  # Coluna alvo

print(f"X: {X.shape}, y:{y.shape}")

# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Normalizando
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

# Classe de rede neural
class NeuralNetworkFromScratch:
    def __init__(self, learning_rate, X_train, y_train, X_test, y_test):
        self.weights = np.random.randn(X_train.shape[1])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.loss_train = []
        self.loss_test = []

    def activation(self, x):
        return 1 / (1 + np.exp(-x))  # Sigmoid
    
    def d_activation(self, x):
        return self.activation(x) * (1 - self.activation(x))  # Derivada da sigmoid

    def forward(self, X):
        hidden = np.dot(X, self.weights) + self.bias
        return self.activation(hidden)

    def backward(self, X, y_true):
        hidden = np.dot(X, self.weights) + self.bias
        y_pred = self.forward(X)

        dL_dypred = 2 * (y_pred - y_true)
        dypred_dz = self.d_activation(hidden)

        dL_db = dL_dypred * dypred_dz * 1
        dL_dw = dL_dypred * dypred_dz * X

        return dL_db, dL_dw

    def optimizer(self, dL_db, dL_dw):
        self.bias -= dL_db * self.learning_rate
        self.weights -= dL_dw * self.learning_rate

    def train(self, ITERATIONS):
        for i in range(ITERATIONS):
            idx = np.random.randint(len(self.X_train))
            x_sample = self.X_train[idx]
            y_sample = self.y_train[idx]

            y_pred = self.forward(x_sample)
            loss = np.square(y_pred - y_sample)
            self.loss_train.append(loss)

            dL_db, dL_dw = self.backward(x_sample, y_sample)
            self.optimizer(dL_db, dL_dw)

            loss_sum = 0
            for j in range(len(self.X_test)):
                y_true = self.y_test[j]
                y_pred = self.forward(self.X_test[j])
                loss_sum += np.square(y_pred - y_true)
            self.loss_test.append(loss_sum)

        return "=== TREINAMENTO CONCLUÍDO ==="

# Hiperparâmetros
learning_rate = 0.1
ITERATIONS = 1000

# Treinamento
nn = NeuralNetworkFromScratch(
    learning_rate=learning_rate,
    X_train=X_train_scale,
    y_train=y_train,
    X_test=X_test_scale,
    y_test=y_test
)

print(nn.train(ITERATIONS=ITERATIONS))
print("Última loss de treino:", nn.loss_train[-1])
print("Última loss de teste:", nn.loss_test[-1])

# Gráfico de evolução da loss de teste
sns.lineplot(x=list(range(len(nn.loss_test))), y=nn.loss_test)
plt.title("Evolução da Loss no Teste")
plt.xlabel("Iteração")
plt.ylabel("Erro Quadrático")
plt.grid(True)
plt.show()

# Avaliação
total = X_test_scale.shape[0]
correct = 0
y_preds = []

for i in range(total):
    y_true = y_test[i]
    y_pred = np.round(nn.forward(X_test_scale[i]))
    y_preds.append(y_pred)
    if y_true == y_pred:
        correct += 1

# Acurácia
accuracy = correct / total
print(f"Acurácia: {accuracy * 100:.2f}%")

# Classe mais comum (baseline)
dist = Counter(map(int, y_test))
print("Distribuição de classes no teste:", dist)

# Matriz de confusão
print("Matriz de Confusão:")
print(confusion_matrix(y_true=y_test, y_pred=y_preds))
