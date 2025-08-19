import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MnistPerceptronSplit {
    public static void main(String[] args) throws Exception {
    // Tamanho do lote para carregar os dados do MNIST
    int batchSize = 128; // Lote maior para carregar tudo em memória
    // Semente para reprodutibilidade dos resultados
    int rngSeed = 12345;
    // Número de épocas de treinamento
    int numEpochs = 10;

    System.out.println("Carregando todo o dataset MNIST na memória...");
        // Objeto que itera sobre o dataset MNIST completo
        DataSetIterator mnistAll = new MnistDataSetIterator(batchSize, false, rngSeed);
        // Lista para armazenar todos os batches do dataset
        List<DataSet> allData = new ArrayList<>();
        int batchCount = 0;
        // Carrega todos os batches do MNIST na lista
        while (mnistAll.hasNext()) {
            DataSet ds = mnistAll.next(); // Objeto DataSet representa um batch de exemplos
            allData.add(ds);
            batchCount++;
            if (batchCount <= 5) {
                System.out.println("Exemplo de batch " + batchCount + ": " + ds.numExamples() + " amostras");
            }
        }
        // Une todos os batches em um único DataSet
        DataSet full = DataSet.merge(allData);
        System.out.println("Embaralhando o dataset...");
        // Embaralha os exemplos para garantir aleatoriedade
        full.shuffle(rngSeed);

    // Divisão dos dados: 60% treino, 20% validação, 20% teste
    int n = full.numExamples(); // Total de exemplos
    int nTrain = (int) (n * 0.6); // Quantidade para treino
    int nVal = (int) (n * 0.2);   // Quantidade para validação
    int nTest = n - nTrain - nVal; // Quantidade para teste
    System.out.println("Total de exemplos: " + n);
    System.out.println("Treino: " + nTrain + ", Validação: " + nVal + ", Teste: " + nTest);

    // Arrays de índices para separar os conjuntos
    int[] idxTrain = new int[nTrain];
    int[] idxVal = new int[nVal];
    int[] idxTest = new int[nTest];
    for (int i = 0; i < nTrain; i++) idxTrain[i] = i;
    for (int i = 0; i < nVal; i++) idxVal[i] = nTrain + i;
    for (int i = 0; i < nTest; i++) idxTest[i] = nTrain + nVal + i;

    System.out.println("Separando dados em treino, validação e teste...");
    // Objetos DataSet para cada conjunto
    DataSet train = new DataSet(full.getFeatures().getRows(idxTrain), full.getLabels().getRows(idxTrain)); // Treinamento
    DataSet val = new DataSet(full.getFeatures().getRows(idxVal), full.getLabels().getRows(idxVal));       // Validação
    DataSet test = new DataSet(full.getFeatures().getRows(idxTest), full.getLabels().getRows(idxTest));   // Teste

    System.out.println("Normalizando dados (MinMaxScaler 0-1)...");
    // Normalizador para escalar os dados entre 0 e 1
    NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler(0, 1);
    scaler.fit(train); // Ajusta o normalizador com base no treino
    scaler.transform(train); // Aplica ao treino
    scaler.transform(val);   // Aplica à validação
    scaler.transform(test);  // Aplica ao teste

    System.out.println("Construindo rede neural: 2 camadas densas (128, 64) + 10 neurônios de saída...");
    // Configuração da rede neural: 2 camadas densas e uma camada de saída
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(rngSeed) // Semente para reprodutibilidade
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // Algoritmo de otimização
        .updater(new Adam(0.001)) // Otimizador Adam com taxa de aprendizado 0.001
        .list()
        // Primeira camada densa: 128 neurônios, ativação ReLU
        .layer(new DenseLayer.Builder().nIn(28*28).nOut(128).activation(Activation.RELU).build())
        // Segunda camada densa: 64 neurônios, ativação ReLU
        .layer(new DenseLayer.Builder().nIn(128).nOut(64).activation(Activation.RELU).build())
        // Camada de saída: 10 neurônios (classes), ativação Softmax
        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .activation(Activation.SOFTMAX).nIn(64).nOut(10).build())
        .build();

    // Cria o modelo de rede neural com a configuração definida
    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init(); // Inicializa os parâmetros da rede
    // Listener para mostrar o score a cada 20 iterações
    model.setListeners(new ScoreIterationListener(20));

        System.out.println("Iniciando treinamento por " + numEpochs + " épocas...");
        // Cria um arquivo CSV para salvar as acurácias por época
        java.io.FileWriter csvWriter = new java.io.FileWriter("acuracia_epocas.csv");
        csvWriter.append("epoca,treino,validacao,teste\n");
        // Loop de treinamento por época
        for (int i = 0; i < numEpochs; i++) {
            System.out.println("Treinando época " + (i+1) + "...");
            model.fit(train); // Treina a rede com o conjunto de treino
            System.out.println("Época " + (i+1) + " finalizada.");

            // Avaliação em treino
            INDArray outTrain = model.output(train.getFeatures()); // Saída da rede para treino
            int acertosTreino = 0;
            int totalTreino = train.numExamples();
            INDArray labelsTreino = train.getLabels(); // Rótulos reais do treino
            for (int j = 0; j < totalTreino; j++) {
                int pred = Nd4j.argMax(outTrain.getRow(j), 0).getInt(0); // Classe prevista
                int real = Nd4j.argMax(labelsTreino.getRow(j), 0).getInt(0); // Classe real
                if (pred == real) acertosTreino++;
            }
            double accTreino = (double) acertosTreino / totalTreino; // Acurácia treino
            System.out.println("Acurácia treino após época " + (i+1) + ": " + accTreino + " (" + String.format("%.2f", accTreino*100) + "%)");

            // Avaliação em validação
            INDArray outVal = model.output(val.getFeatures()); // Saída da rede para validação
            int acertosVal = 0;
            int totalVal = val.numExamples();
            INDArray labelsVal = val.getLabels(); // Rótulos reais da validação
            for (int j = 0; j < totalVal; j++) {
                int pred = Nd4j.argMax(outVal.getRow(j), 0).getInt(0);
                int real = Nd4j.argMax(labelsVal.getRow(j), 0).getInt(0);
                if (pred == real) acertosVal++;
            }
            double accVal = (double) acertosVal / totalVal; // Acurácia validação
            System.out.println("Acurácia validação após época " + (i+1) + ": " + accVal + " (" + String.format("%.2f", accVal*100) + "%)");

            // Avaliação em teste
            INDArray outTest = model.output(test.getFeatures()); // Saída da rede para teste
            int acertosTeste = 0;
            int totalTeste = test.numExamples();
            INDArray labelsTeste = test.getLabels(); // Rótulos reais do teste
            for (int j = 0; j < totalTeste; j++) {
                int pred = Nd4j.argMax(outTest.getRow(j), 0).getInt(0);
                int real = Nd4j.argMax(labelsTeste.getRow(j), 0).getInt(0);
                if (pred == real) acertosTeste++;
            }
            double accTeste = (double) acertosTeste / totalTeste; // Acurácia teste
            System.out.println("Acurácia teste após época " + (i+1) + ": " + accTeste + " (" + String.format("%.2f", accTeste*100) + "%)");

            // Salva os resultados no CSV
            csvWriter.append((i+1) + "," + accTreino + "," + accVal + "," + accTeste + "\n");
        }
        csvWriter.flush(); // Garante que tudo foi escrito
        csvWriter.close(); // Fecha o arquivo

    }
}
