package makina.learn.classification;

import makina.learn.data.DataSet;
import makina.learn.data.DataSetInMemory;
import makina.learn.data.PredictedDataInstance;
import makina.math.matrix.SparseVector;
import makina.math.matrix.Vector;
import makina.math.matrix.VectorType;
import makina.math.matrix.Vectors;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Utilities {
    public static DataSet<PredictedDataInstance<Vector, Double>> parseCovTypeDataFromFile(String filename,
                                                                                          boolean sparseFeatures) {
        String separator = " ";
        List<PredictedDataInstance<Vector, Double>> data = new ArrayList<>();
        try (Stream<String> lines = Files.lines(Paths.get(filename), Charset.defaultCharset())) {
            lines.forEachOrdered(line -> {
                String[] tokens = line.split(separator);
                double label = tokens[0].equals("+1") ? 1 : 0;
                Vector features;
                if (sparseFeatures) {
                    features = Vectors.build(54, VectorType.SPARSE);
                } else {
                    features = Vectors.build(54, VectorType.DENSE);
                }
                for (int i = 1; i < tokens.length; i++) {
                    String[] featurePair = tokens[i].split(":");
                    features.set(Integer.parseInt(featurePair[0]) - 1, Double.parseDouble(featurePair[1]));
                }
                data.add(new PredictedDataInstance<>(null, features, label, null, 1));
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new DataSetInMemory<>(data);
    }

    public static DataSet<PredictedDataInstance<Vector, Double>> parseFisherDataFromFile(String filename) {
        String separator = ",";
        List<PredictedDataInstance<Vector, Double>> data = new ArrayList<>();
        try (Stream<String> lines = Files.lines(Paths.get(filename), Charset.defaultCharset())) {
            lines.forEachOrdered(line -> {
                int numberOfFeatures = line.split(separator).length - 1;
                String[] outputs = line.split(separator);
                SparseVector features = (SparseVector) Vectors.build(numberOfFeatures, VectorType.SPARSE);
                double label = Integer.parseInt(outputs[0]);
                for (int i = 0; i < numberOfFeatures; i++)
                    features.set(i, Double.parseDouble(outputs[i + 1]));
                data.add(new PredictedDataInstance<>(null, features, label, null, 1));
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new DataSetInMemory<>(data);
    }

    public static DataSet<PredictedDataInstance<Vector, Double>> parseURLDataFromFile(String filename,
                                                                                      boolean sparseFeatures) {
        String separator = " ";
        List<PredictedDataInstance<Vector, Double>> data = new ArrayList<>();
        try (Stream<String> lines = Files.lines(Paths.get(filename), Charset.defaultCharset())) {
            lines.limit(30000).forEachOrdered(line -> {
                String[] tokens = line.split(separator);
                double label = tokens[0].equals("+1") ? 1 : 0;
                Vector features;
                if (sparseFeatures) {
                    features = Vectors.build(3231961, VectorType.SPARSE);
                } else {
                    features = Vectors.build(3231961, VectorType.DENSE);
                }
                for (int i = 1; i < tokens.length; i++) {
                    String[] featurePair = tokens[i].split(":");
                    features.set(Integer.parseInt(featurePair[0]) - 1, Double.parseDouble(featurePair[1]));
                }
                data.add(new PredictedDataInstance<>(null, features, label, null, 1));
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new DataSetInMemory<>(data);
    }
}
