import pandas as pd
import numpy as np
from pulizia_dati import carica_e_pulisci
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

df = carica_e_pulisci()
exclude = ['OECD - Total', 'Brazil', 'Russia', 'South Africa', 'Colombia']
df = df.drop(index=[n for n in exclude if n in df.index])

target = 'Life satisfaction'
cols = [
    'Employment rate', 
    'Educational attainment', 
    'Household net adjusted disposable income',
    'Homicide rate', 
    'Self-reported health', 
    'Work-life balance',
    'Quality of support network'
]

features = [f for f in cols if f in df.columns]
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MLPRegressor(
    hidden_layer_sizes=(20, 20), 
    max_iter=10000, 
    alpha=0.5, 
    solver='lbfgs', 
    random_state=42
)

model.fit(X_train_scaled, y_train)

X_all_scaled = scaler.transform(X)
predictions = model.predict(X_all_scaled)

print(f"{'Paese':<25} | {'Reale':<7} | {'Predetto':<8} | {'Err':<6}")
print("-" * 55)

errors = []
for i in range(len(df)):
    err = abs(y.values[i] - predictions[i])
    errors.append(err)
    print(f"{df.index[i]:<25} | {y.values[i]:<7.1f} | {predictions[i]:<8.1f} | {err:<6.2f}")

print("-" * 55)
print(f"MAE: {np.mean(errors):.4f}")

plt.figure(figsize=(10, 7))
plt.scatter(y, predictions, color='blue', edgecolors='black', alpha=0.6)

for i, txt in enumerate(df.index):
    plt.annotate(txt, (y.values[i], predictions[i]), fontsize=7)

plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Reale")
plt.ylabel("Predetto")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()