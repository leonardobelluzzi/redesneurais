# Projeto Redes Neurais - MNIST com Java e DL4J

Este projeto implementa uma rede neural multicamadas (MLP) para classificação de dígitos manuscritos do dataset MNIST utilizando Java e a biblioteca Deeplearning4j (DL4J).

## Estrutura do Projeto

- `src/main/java/`: Código-fonte principal Java
- `dataset/`: Scripts e utilitários para manipulação do MNIST
- `acuracia_epocas.csv`: Resultados de acurácia por época
- `acuracia_epocas.png`: Gráfico da evolução da acurácia
- `plot_acuracia.py`: Script Python para plotar o gráfico

## Como funciona

1. **Carregamento do MNIST**: O código baixa e carrega o dataset MNIST, embaralha e divide em treino (60%), validação (20%) e teste (20%).
2. **Normalização**: Os dados são normalizados para o intervalo [0, 1].
3. **Arquitetura**: Rede neural com duas camadas densas (128 e 64 neurônios) e camada de saída com 10 neurônios (softmax).
4. **Treinamento**: O modelo é treinado por até 150 épocas, mostrando acurácia em treino, validação e teste a cada época.
5. **Resultados**: As acurácias são salvas em CSV e o gráfico é gerado via Python.

## Como rodar

### Requisitos
- Java 8+
- Maven
- Python 3 (para gráficos)
- pandas e matplotlib (para gráficos)

### Passos

1. **Compilar e treinar a rede**

```bash
cd /home/leonardo/dev/neural
mvn compile exec:java -Dexec.mainClass=MnistPerceptronSplit
```

2. **Gerar o gráfico de acurácia**

```bash
source .venv/bin/activate  # se estiver usando venv
python plot_acuracia.py
```

O gráfico será salvo como `acuracia_epocas.png`.

## Personalização
- Altere o número de épocas, arquitetura ou hiperparâmetros no arquivo `MnistPerceptronSplit.java`.
- O script Python pode ser modificado para outros tipos de gráficos ou métricas.

## Créditos
- [Deeplearning4j](https://deeplearning4j.konduit.ai/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

---

**Autor:** Leonardo Belluzzi
