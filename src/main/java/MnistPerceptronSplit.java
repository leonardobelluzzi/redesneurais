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
    int batchSize = 128; // lote maior para carregar tudo em memória
    int rngSeed = 12345;
    int numEpochs = 10;

    System.out.println("Carregando todo o dataset MNIST na memória...");
        DataSetIterator mnistAll = new MnistDataSetIterator(batchSize, false, rngSeed);
        List<DataSet> allData = new ArrayList<>();
        int batchCount = 0;
        while (mnistAll.hasNext()) {
            DataSet ds = mnistAll.next();
            allData.add(ds);
            batchCount++;
            if (batchCount <= 5) {
                System.out.println("Exemplo de batch " + batchCount + ": " + ds.numExamples() + " amostras");
            }
        }
        DataSet full = DataSet.merge(allData);
        System.out.println("Embaralhando o dataset...");
        full.shuffle(rngSeed);

        // Divisão: 60% treino, 20% validação, 20% teste
        int n = full.numExamples();
        int nTrain = (int) (n * 0.6);
        int nVal = (int) (n * 0.2);
        int nTest = n - nTrain - nVal;
        System.out.println("Total de exemplos: " + n);
        System.out.println("Treino: " + nTrain + ", Validação: " + nVal + ", Teste: " + nTest);

        // getRows espera um array de índices, então criamos os arrays
        int[] idxTrain = new int[nTrain];
        int[] idxVal = new int[nVal];
        int[] idxTest = new int[nTest];
        for (int i = 0; i < nTrain; i++) idxTrain[i] = i;
        for (int i = 0; i < nVal; i++) idxVal[i] = nTrain + i;
        for (int i = 0; i < nTest; i++) idxTest[i] = nTrain + nVal + i;

        System.out.println("Separando dados em treino, validação e teste...");
        DataSet train = new DataSet(full.getFeatures().getRows(idxTrain), full.getLabels().getRows(idxTrain));
        DataSet val = new DataSet(full.getFeatures().getRows(idxVal), full.getLabels().getRows(idxVal));
        DataSet test = new DataSet(full.getFeatures().getRows(idxTest), full.getLabels().getRows(idxTest));

        System.out.println("Normalizando dados (MinMaxScaler 0-1)...");
        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler(0, 1);
        scaler.fit(train);
        scaler.transform(train);
        scaler.transform(val);
        scaler.transform(test);

    System.out.println("Construindo rede neural: 2 camadas densas (128, 64) + 10 neurônios de saída...");
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(rngSeed)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(new Adam(0.001))
        .list()
        .layer(new DenseLayer.Builder().nIn(28*28).nOut(128).activation(Activation.RELU).build())
        .layer(new DenseLayer.Builder().nIn(128).nOut(64).activation(Activation.RELU).build())
        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .activation(Activation.SOFTMAX).nIn(64).nOut(10).build())
        .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(20));

        System.out.println("Iniciando treinamento por " + numEpochs + " épocas...");
        // Preparar para salvar acurácias
        java.io.FileWriter csvWriter = new java.io.FileWriter("acuracia_epocas.csv");
        csvWriter.append("epoca,treino,validacao,teste\n");
        for (int i = 0; i < numEpochs; i++) {
            System.out.println("Treinando época " + (i+1) + "...");
            model.fit(train);
            System.out.println("Época " + (i+1) + " finalizada.");

            // Avaliação em treino
            INDArray outTrain = model.output(train.getFeatures());
            int acertosTreino = 0;
            int totalTreino = train.numExamples();
            INDArray labelsTreino = train.getLabels();
            for (int j = 0; j < totalTreino; j++) {
                int pred = Nd4j.argMax(outTrain.getRow(j), 0).getInt(0);
                int real = Nd4j.argMax(labelsTreino.getRow(j), 0).getInt(0);
                if (pred == real) acertosTreino++;
            }
            double accTreino = (double) acertosTreino / totalTreino;
            System.out.println("Acurácia treino após época " + (i+1) + ": " + accTreino + " (" + String.format("%.2f", accTreino*100) + "%)");

            // Avaliação em validação
            INDArray outVal = model.output(val.getFeatures());
            int acertosVal = 0;
            int totalVal = val.numExamples();
            INDArray labelsVal = val.getLabels();
            for (int j = 0; j < totalVal; j++) {
                int pred = Nd4j.argMax(outVal.getRow(j), 0).getInt(0);
                int real = Nd4j.argMax(labelsVal.getRow(j), 0).getInt(0);
                if (pred == real) acertosVal++;
            }
            double accVal = (double) acertosVal / totalVal;
            System.out.println("Acurácia validação após época " + (i+1) + ": " + accVal + " (" + String.format("%.2f", accVal*100) + "%)");

            // Avaliação em teste
            INDArray outTest = model.output(test.getFeatures());
            int acertosTeste = 0;
            int totalTeste = test.numExamples();
            INDArray labelsTeste = test.getLabels();
            for (int j = 0; j < totalTeste; j++) {
                int pred = Nd4j.argMax(outTest.getRow(j), 0).getInt(0);
                int real = Nd4j.argMax(labelsTeste.getRow(j), 0).getInt(0);
                if (pred == real) acertosTeste++;
            }
            double accTeste = (double) acertosTeste / totalTeste;
            System.out.println("Acurácia teste após época " + (i+1) + ": " + accTeste + " (" + String.format("%.2f", accTeste*100) + "%)");

            // Salvar no CSV
            csvWriter.append((i+1) + "," + accTreino + "," + accVal + "," + accTeste + "\n");
        }
        csvWriter.flush();
        csvWriter.close();

    }
}
