import java.awt.image.BufferedImage; // Para criar e manipular imagens
import javax.imageio.ImageIO; // Para salvar imagens em arquivos
import java.io.File; // Para manipulação de arquivos
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator; // Para baixar e iterar sobre o MNIST
import org.nd4j.linalg.dataset.DataSet; // Representa um lote de dados
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator; // Interface para iteradores de datasets
import org.nd4j.linalg.api.ndarray.INDArray; // Estrutura de dados para arrays multidimensionais
import org.nd4j.linalg.indexing.NDArrayIndex; // Para indexação em arrays ND4J

public class MnistDownloadDemo {
    public static void main(String[] args) throws Exception {
        // Define o tamanho do lote (quantas imagens serão carregadas de uma vez)
        int batchSize = 10;
        // Semente para aleatoriedade (reprodutibilidade)
        int rngSeed = 12345;

        // Cria um iterador para o dataset MNIST de treino, baixa os dados se necessário
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);

        // Mensagem informando que o download/carregamento está ocorrendo
        System.out.println("Baixando e carregando MNIST...");
        // Pega o primeiro lote de dados
        DataSet firstBatch = mnistTrain.next();
        // Extrai as imagens (features) e os rótulos (labels) do lote
        INDArray features = firstBatch.getFeatures();
        INDArray labels = firstBatch.getLabels();

        File pasta = new File("imagem");
        if (!pasta.exists()) {
            pasta.mkdir();
        }


        // Salva apenas as 100 primeiras imagens do dataset de treino como arquivos PNG
        int width = 28;
        int height = 28;
        int total = 0;
        int maxToSave = 100;
        
        mnistTrain.reset();
        outer:
        while (mnistTrain.hasNext()) {
            DataSet batch = mnistTrain.next();
            INDArray batchFeatures = batch.getFeatures();
            INDArray batchLabels = batch.getLabels();
            int batchSizeActual = (int) batchFeatures.size(0);
            for (int i = 0; i < batchSizeActual; i++) {
                if (total >= maxToSave) break outer;
                BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
                float[] pixels = batchFeatures.getRow(i).toFloatVector();
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int value = (int) (pixels[y * width + x] * 255);
                        int rgb = (value << 16) | (value << 8) | value;
                        img.setRGB(x, y, rgb);
                    }
                }
                int label = batchLabels.getRow(i).argMax().getInt(0);
                File outputfile = new File(pasta, String.format("mnist_image_%05d_label_%d.png", total, label));
                ImageIO.write(img, "png", outputfile);
                System.out.println(String.format("Imagem %d salva como mnist_image_%05d_label_%d.png", total, total, label));
                total++;
            }
        }
        System.out.println("Total de imagens salvas: " + total);
    }
}
