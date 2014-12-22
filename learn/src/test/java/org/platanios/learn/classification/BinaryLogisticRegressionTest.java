package org.platanios.learn.classification;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.DataSetInMemory;
import org.platanios.learn.data.LabeledDataInstance;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.SparseVector;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorType;
import org.platanios.learn.math.matrix.Vectors;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
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
        DataSet<LabeledDataInstance<Vector, Integer>> trainingDataSet = parseCovTypeDataFromFile(filename, false);
        LogisticRegressionSGD classifier =
                new LogisticRegressionSGD.Builder(trainingDataSet.get(0).features().size())
                        .sparse(false)
                        .build();
        classifier.train(trainingDataSet.subSet(0, 500000));
        DataSet<PredictedDataInstance<Vector, Integer>> testingDataSet = new DataSetInMemory<>();
        for (LabeledDataInstance<Vector, Integer> dataInstance : trainingDataSet.subSet(500000, trainingDataSet.size()))
            testingDataSet.add(new PredictedDataInstance<>(dataInstance.name(),
                                                           dataInstance.features(),
                                                           dataInstance.label(),
                                                           dataInstance.source(),
                                                           1));
        int[] expectedPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < expectedPredictions.length; i++)
            expectedPredictions[i] = testingDataSet.get(i).label();
        DataSet<PredictedDataInstance<Vector, Integer>> predictedDataSet = classifier.predict(testingDataSet);
        int[] actualPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < actualPredictions.length; i++)
            actualPredictions[i] = predictedDataSet.get(i).label();
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testSparseBinaryLogisticRegressionUsingSGD() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        DataSet<LabeledDataInstance<Vector, Integer>> trainingDataSet = parseCovTypeDataFromFile(filename, true);
        LogisticRegressionSGD classifier =
                new LogisticRegressionSGD.Builder(trainingDataSet.get(0).features().size())
                        .sparse(true)
                        .build();
        classifier.train(trainingDataSet.subSet(0, 500000));
        DataSet<PredictedDataInstance<Vector, Integer>> testingDataSet = new DataSetInMemory<>();
        for (LabeledDataInstance<Vector, Integer> dataInstance : trainingDataSet.subSet(500000, trainingDataSet.size()))
            testingDataSet.add(new PredictedDataInstance<>(dataInstance.name(),
                                                           dataInstance.features(),
                                                           dataInstance.label(),
                                                           dataInstance.source(),
                                                           1));
        int[] expectedPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < expectedPredictions.length; i++)
            expectedPredictions[i] = testingDataSet.get(i).label();
        DataSet<PredictedDataInstance<Vector, Integer>> predictedDataSet = classifier.predict(testingDataSet);
        int[] actualPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < actualPredictions.length; i++)
            actualPredictions[i] = predictedDataSet.get(i).label();
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testSmallDenseBinaryLogisticRegressionUsingSGD() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/fisher.binary.txt";
        DataSet<LabeledDataInstance<Vector, Integer>> trainingDataSet = parseFisherDataFromFile(filename);
        LogisticRegressionSGD classifier =
                new LogisticRegressionSGD.Builder(trainingDataSet.get(0).features().size())
                        .sparse(false)
                        .build();
        classifier.train(trainingDataSet.subSet(0, 80));
        DataSet<PredictedDataInstance<Vector, Integer>> testingDataSet = new DataSetInMemory<>();
        for (LabeledDataInstance<Vector, Integer> dataInstance : trainingDataSet.subSet(80, trainingDataSet.size()))
            testingDataSet.add(new PredictedDataInstance<>(dataInstance.name(),
                                                           dataInstance.features(),
                                                           dataInstance.label(),
                                                           dataInstance.source(),
                                                           1));
        int[] expectedPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < expectedPredictions.length; i++)
            expectedPredictions[i] = testingDataSet.get(i).label();
        DataSet<PredictedDataInstance<Vector, Integer>> predictedDataSet = classifier.predict(testingDataSet);
        int[] actualPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < actualPredictions.length; i++)
            actualPredictions[i] = predictedDataSet.get(i).label();
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testDenseBinaryLogisticRegressionUsingAdaGrad() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        DataSet<LabeledDataInstance<Vector, Integer>> trainingDataSet = parseCovTypeDataFromFile(filename, false);
        LogisticRegressionAdaGrad classifier =
                new LogisticRegressionAdaGrad.Builder(trainingDataSet.get(0).features().size())
                        .sparse(false)
                        .build();
        classifier.train(trainingDataSet.subSet(0, 500000));
        DataSet<PredictedDataInstance<Vector, Integer>> testingDataSet = new DataSetInMemory<>();
        for (LabeledDataInstance<Vector, Integer> dataInstance : trainingDataSet.subSet(500000, trainingDataSet.size()))
            testingDataSet.add(new PredictedDataInstance<>(dataInstance.name(),
                                                           dataInstance.features(),
                                                           dataInstance.label(),
                                                           dataInstance.source(),
                                                           1));
        int[] expectedPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < expectedPredictions.length; i++)
            expectedPredictions[i] = testingDataSet.get(i).label();
        DataSet<PredictedDataInstance<Vector, Integer>> predictedDataSet = classifier.predict(testingDataSet);
        int[] actualPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < actualPredictions.length; i++)
            actualPredictions[i] = predictedDataSet.get(i).label();
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testSparseBinaryLogisticRegressionUsingAdaGrad() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        DataSet<LabeledDataInstance<Vector, Integer>> trainingDataSet = parseCovTypeDataFromFile(filename, true);
        LogisticRegressionAdaGrad classifier =
                new LogisticRegressionAdaGrad.Builder(trainingDataSet.get(0).features().size())
                        .batchSize(1000)
                        .maximumNumberOfIterations(1000)
                        .loggingLevel(2)
                        .sparse(true)
                        .build();
        classifier.train(trainingDataSet.subSet(0, 500000));
        DataSet<PredictedDataInstance<Vector, Integer>> testingDataSet = new DataSetInMemory<>();
        for (LabeledDataInstance<Vector, Integer> dataInstance : trainingDataSet.subSet(500000, trainingDataSet.size()))
            testingDataSet.add(new PredictedDataInstance<>(dataInstance.name(),
                                                           dataInstance.features(),
                                                           dataInstance.label(),
                                                           dataInstance.source(),
                                                           1));
        int[] expectedPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < expectedPredictions.length; i++)
            expectedPredictions[i] = testingDataSet.get(i).label();
        DataSet<PredictedDataInstance<Vector, Integer>> predictedDataSet = classifier.predict(testingDataSet);
        int[] actualPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < actualPredictions.length; i++)
            actualPredictions[i] = predictedDataSet.get(i).label();
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testSmallDenseBinaryLogisticRegressionUsingAdaGrad() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/fisher.binary.txt";
        DataSet<LabeledDataInstance<Vector, Integer>> trainingDataSet = parseFisherDataFromFile(filename);
        LogisticRegressionAdaGrad classifier =
                new LogisticRegressionAdaGrad.Builder(trainingDataSet.get(0).features().size())
                        .sparse(false)
                        .loggingLevel(1)
                        .build();
        classifier.train(trainingDataSet.subSet(0, 80));
        DataSet<PredictedDataInstance<Vector, Integer>> testingDataSet = new DataSetInMemory<>();
        for (LabeledDataInstance<Vector, Integer> dataInstance : trainingDataSet.subSet(80, trainingDataSet.size()))
            testingDataSet.add(new PredictedDataInstance<>(dataInstance.name(),
                                                           dataInstance.features(),
                                                           dataInstance.label(),
                                                           dataInstance.source(),
                                                           1));
        int[] expectedPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < expectedPredictions.length; i++)
            expectedPredictions[i] = testingDataSet.get(i).label();
        DataSet<PredictedDataInstance<Vector, Integer>> predictedDataSet = classifier.predict(testingDataSet);
        int[] actualPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < actualPredictions.length; i++)
            actualPredictions[i] = predictedDataSet.get(i).label();
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testLargeSparseBinaryLogisticRegressionUsingAdaGrad() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/url.binary.txt";
        DataSet<LabeledDataInstance<Vector, Integer>> trainingDataSet = parseURLDataFromFile(filename, true);
        LogisticRegressionAdaGrad classifier =
                new LogisticRegressionAdaGrad.Builder(trainingDataSet.get(0).features().size())
                        .sparse(true)
                        .maximumNumberOfIterations(100)
                        .batchSize(1000)
                        .useL1Regularization(true)
                        .l1RegularizationWeight(0.01)
                        .loggingLevel(2)
                        .build();
        classifier.train(trainingDataSet.subSet(0, 20000));
        DataSet<PredictedDataInstance<Vector, Integer>> testingDataSet = new DataSetInMemory<>();
        for (LabeledDataInstance<Vector, Integer> dataInstance : trainingDataSet.subSet(20000, trainingDataSet.size()))
            testingDataSet.add(new PredictedDataInstance<>(dataInstance.name(),
                                                           dataInstance.features(),
                                                           dataInstance.label(),
                                                           dataInstance.source(),
                                                           1));
        int[] expectedPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < expectedPredictions.length; i++)
            expectedPredictions[i] = testingDataSet.get(i).label();
        DataSet<PredictedDataInstance<Vector, Integer>> predictedDataSet = classifier.predict(testingDataSet);
        int[] actualPredictions = new int[testingDataSet.size()];
        double accuracy = 0;
        for (int i = 0; i < actualPredictions.length; i++) {
            actualPredictions[i] = predictedDataSet.get(i).label();
            accuracy += actualPredictions[i] == expectedPredictions[i] ? 1 : 0;
        }
        accuracy /= actualPredictions.length;
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testDenseBinaryLogisticRegressionPrediction() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        DataSet<LabeledDataInstance<Vector, Integer>> trainingDataSet = parseCovTypeDataFromFile(filename, false);
        try {
            filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.dense.model";
            FileInputStream fin = new FileInputStream(filename);
            ObjectInputStream ois = new ObjectInputStream(fin);
            LogisticRegressionPrediction classifier = (LogisticRegressionPrediction) ois.readObject();
            ois.close();
            DataSet<PredictedDataInstance<Vector, Integer>> testingDataSet = new DataSetInMemory<>();
            for (LabeledDataInstance<Vector, Integer> dataInstance : trainingDataSet.subSet(500000, trainingDataSet.size()))
                testingDataSet.add(new PredictedDataInstance<>(dataInstance.name(),
                                                               dataInstance.features(),
                                                               dataInstance.label(),
                                                               dataInstance.source(),
                                                               1));
            int[] expectedPredictions = new int[testingDataSet.size()];
            for (int i = 0; i < expectedPredictions.length; i++)
                expectedPredictions[i] = testingDataSet.get(i).label();
            DataSet<PredictedDataInstance<Vector, Integer>> predictedDataSet = classifier.predict(testingDataSet);
            int[] actualPredictions = new int[testingDataSet.size()];
            for (int i = 0; i < actualPredictions.length; i++)
                actualPredictions[i] = predictedDataSet.get(i).label();
            Assert.assertArrayEquals(expectedPredictions, actualPredictions);
        } catch (IOException|ClassNotFoundException e) {
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void testSparseBinaryLogisticRegressionPrediction() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        DataSet<LabeledDataInstance<Vector, Integer>> trainingDataSet = parseCovTypeDataFromFile(filename, false);
        try {
            filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.sparse.model";
            FileInputStream fin = new FileInputStream(filename);
            ObjectInputStream ois = new ObjectInputStream(fin);
            LogisticRegressionPrediction classifier = (LogisticRegressionPrediction) ois.readObject();
            ois.close();
            DataSet<PredictedDataInstance<Vector, Integer>> testingDataSet = new DataSetInMemory<>();
            for (LabeledDataInstance<Vector, Integer> dataInstance : trainingDataSet.subSet(500000, trainingDataSet.size()))
                testingDataSet.add(new PredictedDataInstance<>(dataInstance.name(),
                                                               dataInstance.features(),
                                                               dataInstance.label(),
                                                               dataInstance.source(),
                                                               1));
            int[] expectedPredictions = new int[testingDataSet.size()];
            for (int i = 0; i < expectedPredictions.length; i++)
                expectedPredictions[i] = testingDataSet.get(i).label();
            DataSet<PredictedDataInstance<Vector, Integer>> predictedDataSet = classifier.predict(testingDataSet);
            int[] actualPredictions = new int[testingDataSet.size()];
            for (int i = 0; i < actualPredictions.length; i++)
                actualPredictions[i] = predictedDataSet.get(i).label();
            Assert.assertArrayEquals(expectedPredictions, actualPredictions);
        } catch (IOException|ClassNotFoundException e) {
            Assert.fail(e.getMessage());
        }
    }

    public static DataSet<LabeledDataInstance<Vector, Integer>> parseCovTypeDataFromFile(String filename,
                                                                                         boolean sparseFeatures) {
        String separator = " ";
        List<LabeledDataInstance<Vector, Integer>> data = new ArrayList<>();
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
                data.add(new LabeledDataInstance<>(null, features, label, null));
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new DataSetInMemory<>(data);
    }

    public static DataSet<LabeledDataInstance<Vector, Integer>> parseFisherDataFromFile(String filename) {
        String separator = ",";
        List<LabeledDataInstance<Vector, Integer>> data = new ArrayList<>();
        try (Stream<String> lines = Files.lines(Paths.get(filename), Charset.defaultCharset())) {
            lines.forEachOrdered(line -> {
                int numberOfFeatures = line.split(separator).length - 1;
                String[] outputs = line.split(separator);
                SparseVector features = (SparseVector) Vectors.build(numberOfFeatures, VectorType.SPARSE);
                int label = Integer.parseInt(outputs[0]);
                for (int i = 0; i < numberOfFeatures; i++)
                    features.set(i, Double.parseDouble(outputs[i + 1]));
                data.add(new LabeledDataInstance<>(null, features, label, null));
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new DataSetInMemory<>(data);
    }

    public static DataSet<LabeledDataInstance<Vector, Integer>> parseURLDataFromFile(String filename,
                                                                                     boolean sparseFeatures) {
        String separator = " ";
        List<LabeledDataInstance<Vector, Integer>> data = new ArrayList<>();
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
                data.add(new LabeledDataInstance<>(null, features, label, null));
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new DataSetInMemory<>(data);
    }
}
