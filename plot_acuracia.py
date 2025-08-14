import pandas as pd
import matplotlib.pyplot as plt

# Lê o CSV gerado pelo treinamento
csv_path = 'acuracia_epocas.csv'
df = pd.read_csv(csv_path)

plt.figure(figsize=(12,7))
plt.plot(df['epoca'], df['acuracia_treino_percent'], label='Treino (%)', color='blue')
plt.plot(df['epoca'], df['acuracia_validacao_percent'], label='Validação (%)', color='orange')
plt.plot(df['epoca'], df['acuracia_teste_percent'], label='Teste (%)', color='green')
plt.xlabel('Época')
plt.ylabel('Acurácia (%)')
plt.title('Acurácia por Época - Treino, Validação e Teste')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('acuracia_epocas.png')
plt.show()
