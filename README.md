# Stock Market with Python and Tensor Flow.
Cria um modelo que prevê a entrada no mercado de ações brasileiro usando Python, Machine Learning e TensorFlow.

Para criar um modelo que prevê a entrada no mercado de ações brasileiro usando Python, Machine Learning e TensorFlow, podemos seguir um processo similar ao anterior, mas utilizando redes neurais através do TensorFlow. Aqui está um exemplo passo a passo:

### Passo 1: Coleta de Dados

Vamos coletar dados históricos de uma ação brasileira, por exemplo, Petrobras (PETR4.SA).

```python
!pip install yfinance pandas tensorflow scikit-learn

import yfinance as yf
import pandas as pd

# Coletar dados de uma ação específica
ticker = 'PETR4.SA'  # Exemplo: Petrobras
dados = yf.download(ticker, start='2020-01-01', end='2023-01-01')

# Visualizar os dados
print(dados.head())
```

### Passo 2: Pré-processamento dos Dados

Vamos preparar os dados para treinar a rede neural.

```python
# Remover valores nulos
dados = dados.dropna()

# Adicionar médias móveis
dados['MA20'] = dados['Close'].rolling(window=20).mean()
dados['MA50'] = dados['Close'].rolling(window=50).mean()

# Adicionar mudança percentual diária
dados['Daily Return'] = dados['Close'].pct_change()

# Remover valores nulos gerados pelas médias móveis e mudança percentual
dados = dados.dropna()

print(dados.head())
```

### Passo 3: Seleção de Recursos

Selecionar as características (features) que serão usadas para treinar o modelo.

```python
features = ['Close', 'Volume', 'MA20', 'MA50', 'Daily Return']
target = 'Close'

X = dados[features]
y = dados[target]
```

### Passo 4: Normalização dos Dados

Normalizar os dados para melhorar o desempenho da rede neural.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled[:5])
```

### Passo 5: Divisão dos Dados

Dividir os dados em conjuntos de treino e teste.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

### Passo 6: Treinamento do Modelo com TensorFlow

Criar e treinar um modelo de rede neural com TensorFlow.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Definir a arquitetura da rede neural
modelo = Sequential()
modelo.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
modelo.add(Dense(32, activation='relu'))
modelo.add(Dense(1))

# Compilar o modelo
modelo.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
historia = modelo.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
```

### Passo 7: Avaliação do Modelo

Avaliar a performance do modelo com os dados de teste.

```python
# Fazer previsões
predicoes = modelo.predict(X_test)

# Avaliar o modelo
mae = tf.keras.metrics.mean_absolute_error(y_test, predicoes).numpy()
mse = tf.keras.metrics.mean_squared_error(y_test, predicoes).numpy()
rmse = mse ** 0.5

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
```

### Passo 8: Predição

Usar o modelo para fazer previsões futuras.

```python
# Fazer uma previsão com novos dados (por exemplo, o último dia do conjunto de dados)
nova_predicao = modelo.predict([X_scaled[-1]])
print(f'Predição do preço de fechamento: {nova_predicao[0][0]}')
```

Este é um exemplo básico de como criar um modelo de previsão de preços de ações com TensorFlow. Para melhorar o desempenho, considere ajustar os hiperparâmetros, adicionar mais camadas à rede neural ou utilizar técnicas de regularização.
