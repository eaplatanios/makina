package org.platanios.learn.classification;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.DataSetInMemory;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.SparseVector;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorType;
import org.platanios.learn.math.matrix.Vectors;

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
public class CrossValidationTrainingTest {
    @Test
    public void testGridSearchFisherDataSet() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/fisher.binary.txt";
        DataSet<PredictedDataInstance<Vector, Integer>> trainingDataSet = parseFisherDataFromFile(filename);
        LogisticRegressionAdaGrad.Builder classifierBuilder =
                new LogisticRegressionAdaGrad.Builder(trainingDataSet.get(0).features().size())
                        .sparse(false)
                        .useL1Regularization(true)
                        .useL2Regularization(true)
                        .maximumNumberOfIterations(1000)
                        .maximumNumberOfIterationsWithNoPointChange(10)
                        .loggingLevel(0);
        CrossValidationTraining<Vector, Integer> training =
                new CrossValidationTraining.Builder<>(classifierBuilder, trainingDataSet.subSet(0, 80))
                        .numberOfFolds(10)
                        .addAllowedParameterValues("sampleWithReplacement", true, false)
                        .addAllowedParameterValues("batchSize", 1, 10, 20)
                        .addAllowedParameterValues("useBiasTerm", true, false )
                        .addAllowedParameterValues("l1RegularizationWeight", 0.1, 1.0, 10.0)
                        .addAllowedParameterValues("l2RegularizationWeight", 0.1, 1.0, 10.0)
                        .build();
        LogisticRegressionAdaGrad classifier = (LogisticRegressionAdaGrad) training.train();
        DataSet<PredictedDataInstance<Vector, Integer>> testingDataSet = new DataSetInMemory<>();
        for (PredictedDataInstance<Vector, Integer> dataInstance : trainingDataSet.subSet(80, trainingDataSet.size()))
            testingDataSet.add(dataInstance);
        int[] expectedPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < expectedPredictions.length; i++)
            expectedPredictions[i] = testingDataSet.get(i).label();
        DataSet<PredictedDataInstance<Vector, Integer>> predictedDataSet = classifier.predictInPlace(testingDataSet);
        int[] actualPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < actualPredictions.length; i++)
            actualPredictions[i] = predictedDataSet.get(i).label();
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testGridSearchCovTypeDataSet() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        DataSet<PredictedDataInstance<Vector, Integer>> trainingDataSet = parseCovTypeDataFromFile(filename, true);
        LogisticRegressionAdaGrad.Builder classifierBuilder =
                new LogisticRegressionAdaGrad.Builder(trainingDataSet.get(0).features().size())
                        .maximumNumberOfIterations(1000)
                        .loggingLevel(0)
                        .sparse(true);
        CrossValidationTraining<Vector, Integer> training =
                new CrossValidationTraining.Builder<>(classifierBuilder, trainingDataSet.subSet(0, 500000))
                        .numberOfFolds(10)
                        .addAllowedParameterValues("sampleWithReplacement", true, false)
                        .addAllowedParameterValues("batchSize", 1, 10, 20)
                        .addAllowedParameterValues("useBiasTerm", true, false )
                        .addAllowedParameterValues("l1RegularizationWeight", 0.1, 1.0, 10.0)
                        .addAllowedParameterValues("l2RegularizationWeight", 0.1, 1.0, 10.0)
                        .build();
        LogisticRegressionAdaGrad classifier = (LogisticRegressionAdaGrad) training.train();
        DataSet<PredictedDataInstance<Vector, Integer>> testingDataSet = new DataSetInMemory<>();
        for (PredictedDataInstance<Vector, Integer> dataInstance : trainingDataSet.subSet(500000, trainingDataSet.size()))
            testingDataSet.add(dataInstance);
        int[] expectedPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < expectedPredictions.length; i++)
            expectedPredictions[i] = testingDataSet.get(i).label();
        DataSet<PredictedDataInstance<Vector, Integer>> predictedDataSet = classifier.predictInPlace(testingDataSet);
        int[] actualPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < actualPredictions.length; i++)
            actualPredictions[i] = predictedDataSet.get(i).label();
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testGridSearchURLDataSet() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/url.binary.txt";
        DataSet<PredictedDataInstance<Vector, Integer>> trainingDataSet = parseURLDataFromFile(filename, true);
        LogisticRegressionAdaGrad.Builder classifierBuilder =
                new LogisticRegressionAdaGrad.Builder(trainingDataSet.get(0).features().size())
                        .sparse(true)
                        .useL1Regularization(true)
                        .useL2Regularization(true)
                        .maximumNumberOfIterations(1000)
                        .maximumNumberOfIterationsWithNoPointChange(10)
                        .loggingLevel(0);
        CrossValidationTraining<Vector, Integer> training =
                new CrossValidationTraining.Builder<>(classifierBuilder, trainingDataSet.subSet(0, 20000))
                        .numberOfFolds(10)
                        .addAllowedParameterValues("sampleWithReplacement", true, false)
                        .addAllowedParameterValues("batchSize", 1, 10)
                        .addAllowedParameterValues("useBiasTerm", true, false )
                        .addAllowedParameterValues("l1RegularizationWeight", 0.1, 1.0, 10.0)
                        .addAllowedParameterValues("l2RegularizationWeight", 0.1, 1.0, 10.0)
                        .build();
        LogisticRegressionAdaGrad classifier = (LogisticRegressionAdaGrad) training.train();
        DataSet<PredictedDataInstance<Vector, Integer>> testingDataSet = new DataSetInMemory<>();
        for (PredictedDataInstance<Vector, Integer> dataInstance : trainingDataSet.subSet(20000, trainingDataSet.size()))
            testingDataSet.add(dataInstance);
        int[] expectedPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < expectedPredictions.length; i++)
            expectedPredictions[i] = testingDataSet.get(i).label();
        DataSet<PredictedDataInstance<Vector, Integer>> predictedDataSet = classifier.predictInPlace(testingDataSet);
        int[] actualPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < actualPredictions.length; i++)
            actualPredictions[i] = predictedDataSet.get(i).label();
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    public static DataSet<PredictedDataInstance<Vector, Integer>> parseCovTypeDataFromFile(String filename,
                                                                                           boolean sparseFeatures) {
        String separator = " ";
        List<PredictedDataInstance<Vector, Integer>> data = new ArrayList<>();
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
                data.add(new PredictedDataInstance<>(null, features, label, null, 1));
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new DataSetInMemory<>(data);
    }

    public static DataSet<PredictedDataInstance<Vector, Integer>> parseFisherDataFromFile(String filename) {
        String separator = ",";
        List<PredictedDataInstance<Vector, Integer>> data = new ArrayList<>();
        try (Stream<String> lines = Files.lines(Paths.get(filename), Charset.defaultCharset())) {
            lines.forEachOrdered(line -> {
                int numberOfFeatures = line.split(separator).length - 1;
                String[] outputs = line.split(separator);
                SparseVector features = (SparseVector) Vectors.build(numberOfFeatures, VectorType.SPARSE);
                int label = Integer.parseInt(outputs[0]);
                for (int i = 0; i < numberOfFeatures; i++)
                    features.set(i, Double.parseDouble(outputs[i + 1]));
                data.add(new PredictedDataInstance<>(null, features, label, null, 1));
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new DataSetInMemory<>(data);
    }

    public static DataSet<PredictedDataInstance<Vector, Integer>> parseURLDataFromFile(String filename,
                                                                                       boolean sparseFeatures) {
        String separator = " ";
        List<PredictedDataInstance<Vector, Integer>> data = new ArrayList<>();
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
                data.add(new PredictedDataInstance<>(null, features, label, null, 1));
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new DataSetInMemory<>(data);
    }
}
