import pandas as pd
import matplotlib.pyplot as plt

# Lê o CSV gerado pelo treinamento
csv_path = 'acuracia_epocas.csv'
df = pd.read_csv(csv_path)

plt.figure(figsize=(12,7))
plt.plot(df['epoca'], df['treino']*100, label='Treino (%)', color='blue')
plt.plot(df['epoca'], df['validacao']*100, label='Validação (%)', color='orange')
plt.plot(df['epoca'], df['teste']*100, label='Teste (%)', color='green')
plt.xlabel('Época')
plt.ylabel('Acurácia (%)')
plt.title('Acurácia por Época - Treino, Validação e Teste')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('acuracia_epocas.png')
plt.show()
