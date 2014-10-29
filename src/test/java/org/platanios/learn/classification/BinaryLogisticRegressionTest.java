package org.platanios.learn.classification;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.math.matrix.HashVector;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.math.matrix.VectorType;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * @author Emmanouil Antonios Platanios
 */
public class BinaryLogisticRegressionTest {
    @Test
    public void testDenseBinaryLogisticRegressionUsingSGD() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        List<DataInstance<Vector, Integer>> data = parseCovTypeDataFromFile(filename, false);
        LogisticRegressionSGD classifier =
                new LogisticRegressionSGD.Builder(data.get(0).features().size())
                        .sparse(false)
                        .build();
        classifier.train(data.subList(0, 500000));
        double[] actualPredictionsProbabilities = classifier.predict(data.subList(500000, data.size()));
        int[] actualPredictions = new int[actualPredictionsProbabilities.length];
        int[] expectedPredictions = new int[actualPredictionsProbabilities.length];
        for (int i = 0; i < actualPredictions.length; i++) {
            actualPredictions[i] = actualPredictionsProbabilities[i] >= 0.5 ? 1 : 0;
            expectedPredictions[i] = data.get(500000 + i).label();
        }
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testSparseBinaryLogisticRegressionUsingSGD() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        List<DataInstance<Vector, Integer>> data = parseCovTypeDataFromFile(filename, true);
        LogisticRegressionSGD classifier =
                new LogisticRegressionSGD.Builder(data.get(0).features().size())
                        .sparse(true)
                        .build();
        classifier.train(data.subList(0, 500000));
        double[] actualPredictionsProbabilities = classifier.predict(data.subList(500000, data.size()));
        int[] actualPredictions = new int[actualPredictionsProbabilities.length];
        int[] expectedPredictions = new int[actualPredictionsProbabilities.length];
        for (int i = 0; i < actualPredictions.length; i++) {
            actualPredictions[i] = actualPredictionsProbabilities[i] >= 0.5 ? 1 : 0;
            expectedPredictions[i] = data.get(500000 + i).label();
        }
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testSmallDenseBinaryLogisticRegressionUsingSGD() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/fisher.binary.txt";
        List<DataInstance<Vector, Integer>> data = parseFisherDataFromFile(filename);
        LogisticRegressionSGD classifier =
                new LogisticRegressionSGD.Builder(data.get(0).features().size())
                        .sparse(false)
                        .build();
        classifier.train(data.subList(0, 80));
        double[] actualPredictionsProbabilities = classifier.predict(data.subList(80, data.size()));
        int[] actualPredictions = new int[actualPredictionsProbabilities.length];
        int[] expectedPredictions = new int[actualPredictionsProbabilities.length];
        for (int i = 0; i < actualPredictions.length; i++) {
            actualPredictions[i] = actualPredictionsProbabilities[i] >= 0.5 ? 1 : 0;
            expectedPredictions[i] = data.get(80 + i).label();
        }
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testDenseBinaryLogisticRegressionUsingAdaGrad() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        List<DataInstance<Vector, Integer>> data = parseCovTypeDataFromFile(filename, false);
        LogisticRegressionAdaGrad classifier =
                new LogisticRegressionAdaGrad.Builder(data.get(0).features().size())
                        .sparse(false)
                        .build();
        classifier.train(data.subList(0, 500000));
        double[] actualPredictionsProbabilities = classifier.predict(data.subList(500000, data.size()));
        int[] actualPredictions = new int[actualPredictionsProbabilities.length];
        int[] expectedPredictions = new int[actualPredictionsProbabilities.length];
        for (int i = 0; i < actualPredictions.length; i++) {
            actualPredictions[i] = actualPredictionsProbabilities[i] >= 0.5 ? 1 : 0;
            expectedPredictions[i] = data.get(500000 + i).label();
        }
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testSparseBinaryLogisticRegressionUsingAdaGrad() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        List<DataInstance<Vector, Integer>> data = parseCovTypeDataFromFile(filename, true);
        LogisticRegressionAdaGrad classifier =
                new LogisticRegressionAdaGrad.Builder(data.get(0).features().size())
                        .batchSize(1000)
                        .maximumNumberOfIterations(1000)
                        .loggingLevel(2)
                        .sparse(true)
                        .build();
        classifier.train(data.subList(0, 500000));
        double[] actualPredictionsProbabilities = classifier.predict(data.subList(500000, data.size()));
        int[] actualPredictions = new int[actualPredictionsProbabilities.length];
        int[] expectedPredictions = new int[actualPredictionsProbabilities.length];
        for (int i = 0; i < actualPredictions.length; i++) {
            actualPredictions[i] = actualPredictionsProbabilities[i] >= 0.5 ? 1 : 0;
            expectedPredictions[i] = data.get(500000 + i).label();
        }
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testSmallDenseBinaryLogisticRegressionUsingAdaGrad() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/fisher.binary.txt";
        List<DataInstance<Vector, Integer>> data = parseFisherDataFromFile(filename);
        LogisticRegressionAdaGrad classifier =
                new LogisticRegressionAdaGrad.Builder(data.get(0).features().size())
                        .sparse(false)
                        .build();
        classifier.train(data.subList(0, 80));
        double[] actualPredictionsProbabilities = classifier.predict(data.subList(80, data.size()));
        int[] actualPredictions = new int[actualPredictionsProbabilities.length];
        int[] expectedPredictions = new int[actualPredictionsProbabilities.length];
        for (int i = 0; i < actualPredictions.length; i++) {
            actualPredictions[i] = actualPredictionsProbabilities[i] >= 0.5 ? 1 : 0;
            expectedPredictions[i] = data.get(80 + i).label();
        }
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testLargeSparseBinaryLogisticRegressionUsingAdaGrad() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/url.binary.txt";
        List<DataInstance<Vector, Integer>> data = parseURLDataFromFile(filename, true);
        LogisticRegressionAdaGrad classifier =
                new LogisticRegressionAdaGrad.Builder(data.get(0).features().size())
                        .sparse(true)
                        .maximumNumberOfIterations(100)
                        .batchSize(1000)
                        .useL1Regularization(true)
                        .l1RegularizationWeight(0.01)
                        .loggingLevel(2)
                        .build();
        classifier.train(data.subList(0, 20000));
        double[] actualPredictionsProbabilities = classifier.predict(data.subList(20000, data.size()));
        int[] actualPredictions = new int[actualPredictionsProbabilities.length];
        int[] expectedPredictions = new int[actualPredictionsProbabilities.length];
        double accuracy = 0;
        for (int i = 0; i < actualPredictions.length; i++) {
            actualPredictions[i] = actualPredictionsProbabilities[i] >= 0.5 ? 1 : 0;
            expectedPredictions[i] = data.get(20000 + i).label();
            accuracy += actualPredictions[i] == expectedPredictions[i] ? 1 : 0;
        }
        accuracy /= actualPredictions.length;
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testDenseBinaryLogisticRegressionPrediction() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        List<DataInstance<Vector, Integer>> data = parseCovTypeDataFromFile(filename, false);
        try {
            filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.dense.model";
            FileInputStream fin = new FileInputStream(filename);
            ObjectInputStream ois = new ObjectInputStream(fin);
            LogisticRegressionPrediction classifier =
                    new LogisticRegressionPrediction.Builder(ois).build();
            ois.close();
            double[] actualPredictionsProbabilities = classifier.predict(data.subList(500000, data.size()));
            int[] actualPredictions = new int[actualPredictionsProbabilities.length];
            int[] expectedPredictions = new int[actualPredictionsProbabilities.length];
            for (int i = 0; i < actualPredictions.length; i++) {
                actualPredictions[i] = actualPredictionsProbabilities[i] >= 0.5 ? 1 : 0;
                expectedPredictions[i] = data.get(500000 + i).label();
            }
            Assert.assertArrayEquals(expectedPredictions, actualPredictions);
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void testSparseBinaryLogisticRegressionPrediction() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        List<DataInstance<Vector, Integer>> data = parseCovTypeDataFromFile(filename, false);
        try {
            filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.sparse.model";
            FileInputStream fin = new FileInputStream(filename);
            ObjectInputStream ois = new ObjectInputStream(fin);
            LogisticRegressionPrediction classifier =
                    new LogisticRegressionPrediction.Builder(ois).build();
            ois.close();
            double[] actualPredictionsProbabilities = classifier.predict(data.subList(500000, data.size()));
            int[] actualPredictions = new int[actualPredictionsProbabilities.length];
            int[] expectedPredictions = new int[actualPredictionsProbabilities.length];
            for (int i = 0; i < actualPredictions.length; i++) {
                actualPredictions[i] = actualPredictionsProbabilities[i] >= 0.5 ? 1 : 0;
                expectedPredictions[i] = data.get(500000 + i).label();
            }
            Assert.assertArrayEquals(expectedPredictions, actualPredictions);
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }

    public static List<DataInstance<Vector, Integer>> parseCovTypeDataFromFile(String filename,
                                                                               boolean sparseFeatures) {
        String separator = " ";
        List<DataInstance<Vector, Integer>> data = new ArrayList<>();
        try (Stream<String> lines = Files.lines(Paths.get(filename), Charset.defaultCharset())) {
            lines.forEachOrdered(line -> {
                String[] tokens = line.split(separator);
                int label = tokens[0].equals("+1") ? 1 : 0;
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
                data.add(new DataInstance.Builder<Vector, Integer>(features).label(label).build());
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    public static List<DataInstance<Vector, Integer>> parseFisherDataFromFile(String filename) {
        String separator = ",";
        List<DataInstance<Vector, Integer>> data = new ArrayList<>();
        try (Stream<String> lines = Files.lines(Paths.get(filename), Charset.defaultCharset())) {
            lines.forEachOrdered(line -> {
                int numberOfFeatures = line.split(separator).length - 1;
                String[] outputs = line.split(separator);
                HashVector features = (HashVector) Vectors.build(numberOfFeatures, VectorType.SPARSE);
                int label = Integer.parseInt(outputs[0]);
                for (int i = 0; i < numberOfFeatures; i++) {
                    features.set(i, Double.parseDouble(outputs[i + 1]));
                }
                data.add(new DataInstance.Builder<Vector, Integer>(features).label(label).build());
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    public static List<DataInstance<Vector, Integer>> parseURLDataFromFile(String filename,
                                                                           boolean sparseFeatures) {
        String separator = " ";
        List<DataInstance<Vector, Integer>> data = new ArrayList<>();
        try (Stream<String> lines = Files.lines(Paths.get(filename), Charset.defaultCharset())) {
            lines.limit(30000).forEachOrdered(line -> {
                String[] tokens = line.split(separator);
                int label = tokens[0].equals("+1") ? 1 : 0;
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
                data.add(new DataInstance.Builder<Vector, Integer>(features).label(label).build());
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }
}
