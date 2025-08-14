import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;

public class MnistExample {
    public static void main(String[] args) throws Exception {
        int batchSize = 64;
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);
        System.out.println("Primeiro batch: " + mnistTrain.next().numExamples());
    }
}
