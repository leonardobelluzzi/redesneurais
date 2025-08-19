# Projeto Redes Neurais - MNIST com Java e DL4J

Este projeto implementa uma rede neural multicamadas (MLP) para classificação de dígitos manuscritos do dataset MNIST utilizando Java e a biblioteca Deeplearning4j (DL4J).

## Histórico de Atualizações

- **Adição de comentários detalhados no código Java**: O arquivo `MnistPerceptronSplit.java` agora possui comentários explicando cada objeto, variável e etapa do processo, facilitando o entendimento para novos usuários.
- **Ajuste da arquitetura da rede**: Foram realizados testes com diferentes quantidades de camadas e neurônios, permitindo ao usuário personalizar facilmente a arquitetura no código.
- **Padronização do número de épocas**: O número de épocas de treinamento pode ser ajustado diretamente no início do arquivo Java, tornando a experimentação mais simples.
- **Geração automática de CSV**: A cada execução do treinamento, um novo arquivo `acuracia_epocas.csv` é gerado, contendo as acurácias de treino, validação e teste por época.
- **Script Python para gráficos**: O arquivo `plot_acuracia.py` foi criado para ler o CSV e gerar automaticamente o gráfico de evolução da acurácia.

## Estrutura do Projeto

- `src/main/java/`: Código-fonte principal Java
- `dataset/`: Scripts e utilitários para manipulação do MNIST
- `acuracia_epocas.csv`: Resultados de acurácia por época
- `acuracia_epocas.png`: Gráfico da evolução da acurácia
- `plot_acuracia.py`: Script Python para plotar o gráfico

## Estrutura do Projeto

- `src/main/java/`: Código-fonte principal Java, incluindo o arquivo `MnistPerceptronSplit.java`.
- `dataset/`: Imagens do MNIST.
- `acuracia_epocas.csv`: Resultados de acurácia por época, gerado automaticamente após cada treinamento.
- `acuracia_epocas.png`: Gráfico da evolução da acurácia, gerado pelo script Python.
- `plot_acuracia.py`: Script Python para plotar o gráfico a partir do CSV.

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


**Autor:** Leonardo Belluzzi
