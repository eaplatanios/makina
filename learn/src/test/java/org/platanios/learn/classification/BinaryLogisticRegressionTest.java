package org.platanios.learn.classification;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.DataSetInMemory;
import org.platanios.learn.data.LabeledDataInstance;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.math.matrix.Vector;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * @author Emmanouil Antonios Platanios
 */
public class BinaryLogisticRegressionTest {
    @Test
    public void testDenseBinaryLogisticRegressionUsingSGD() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        DataSet<PredictedDataInstance<Vector, Double>> trainingDataSet = Utilities.parseCovTypeDataFromFile(filename, false);
        LogisticRegressionSGD classifier =
                new LogisticRegressionSGD.Builder(trainingDataSet.get(0).features().size())
                        .sparse(false)
                        .build();
        classifier.train(trainingDataSet.subSet(0, 500000));
        DataSet<PredictedDataInstance<Vector, Double>> testingDataSet = new DataSetInMemory<>();
        for (LabeledDataInstance<Vector, Double> dataInstance : trainingDataSet.subSet(500000, trainingDataSet.size()))
            testingDataSet.add(new PredictedDataInstance<>(dataInstance.name(),
                                                           dataInstance.features(),
                                                           dataInstance.label(),
                                                           dataInstance.source(),
                                                           1));
        int[] expectedPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < expectedPredictions.length; i++)
            expectedPredictions[i] = (int) (double) testingDataSet.get(i).label();
        DataSet<PredictedDataInstance<Vector, Double>> predictedDataSet = classifier.predictInPlace(testingDataSet);
        int[] actualPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < actualPredictions.length; i++)
            actualPredictions[i] = (int) (double) predictedDataSet.get(i).label();
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testSparseBinaryLogisticRegressionUsingSGD() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        DataSet<PredictedDataInstance<Vector, Double>> trainingDataSet = Utilities.parseCovTypeDataFromFile(filename, true);
        LogisticRegressionSGD classifier =
                new LogisticRegressionSGD.Builder(trainingDataSet.get(0).features().size())
                        .sparse(true)
                        .build();
        classifier.train(trainingDataSet.subSet(0, 500000));
        DataSet<PredictedDataInstance<Vector, Double>> testingDataSet = new DataSetInMemory<>();
        for (LabeledDataInstance<Vector, Double> dataInstance : trainingDataSet.subSet(500000, trainingDataSet.size()))
            testingDataSet.add(new PredictedDataInstance<>(dataInstance.name(),
                                                           dataInstance.features(),
                                                           dataInstance.label(),
                                                           dataInstance.source(),
                                                           1));
        int[] expectedPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < expectedPredictions.length; i++)
            expectedPredictions[i] = (int) (double) testingDataSet.get(i).label();
        DataSet<PredictedDataInstance<Vector, Double>> predictedDataSet = classifier.predictInPlace(testingDataSet);
        int[] actualPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < actualPredictions.length; i++)
            actualPredictions[i] = (int) (double) predictedDataSet.get(i).label();
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testSmallDenseBinaryLogisticRegressionUsingSGDAndDataSetWithPredictedDataInstances() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/fisher.binary.txt";
        DataSet<PredictedDataInstance<Vector, Double>> trainingDataSet = Utilities.parseFisherDataFromFile(filename);
        LogisticRegressionSGD classifier =
                new LogisticRegressionSGD.Builder(trainingDataSet.get(0).features().size())
                        .sparse(false)
                        .build();
        classifier.train(trainingDataSet.subSet(0, 80));
        DataSet<PredictedDataInstance<Vector, Double>> testingDataSet = new DataSetInMemory<>();
        for (PredictedDataInstance<Vector, Double> dataInstance : trainingDataSet.subSet(80, trainingDataSet.size()))
            testingDataSet.add(dataInstance);
        int[] expectedPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < expectedPredictions.length; i++)
            expectedPredictions[i] = (int) (double) testingDataSet.get(i).label();
        DataSet<PredictedDataInstance<Vector, Double>> predictedDataSet = classifier.predictInPlace(testingDataSet);
        int[] actualPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < actualPredictions.length; i++)
            actualPredictions[i] = (int) (double) predictedDataSet.get(i).label();
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testDenseBinaryLogisticRegressionUsingAdaGrad() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        DataSet<PredictedDataInstance<Vector, Double>> trainingDataSet = Utilities.parseCovTypeDataFromFile(filename, false);
        LogisticRegressionAdaGrad classifier =
                new LogisticRegressionAdaGrad.Builder(trainingDataSet.get(0).features().size())
                        .sparse(false)
                        .build();
        classifier.train(trainingDataSet.subSet(0, 500000));
        DataSet<PredictedDataInstance<Vector, Double>> testingDataSet = new DataSetInMemory<>();
        for (LabeledDataInstance<Vector, Double> dataInstance : trainingDataSet.subSet(500000, trainingDataSet.size()))
            testingDataSet.add(new PredictedDataInstance<>(dataInstance.name(),
                                                           dataInstance.features(),
                                                           dataInstance.label(),
                                                           dataInstance.source(),
                                                           1));
        int[] expectedPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < expectedPredictions.length; i++)
            expectedPredictions[i] = (int) (double) testingDataSet.get(i).label();
        DataSet<PredictedDataInstance<Vector, Double>> predictedDataSet = classifier.predictInPlace(testingDataSet);
        int[] actualPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < actualPredictions.length; i++)
            actualPredictions[i] = (int) (double) predictedDataSet.get(i).label();
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

//    @Test
//    public void testSparseBinaryLogisticRegressionUsingAdaGrad() {
//        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
//        DataSet<PredictedDataInstance<Vector, Double>> trainingDataSet = Utilities.parseCovTypeDataFromFile(filename, true);
//        LogisticRegressionAdaGrad classifier =
//                new LogisticRegressionAdaGrad.Builder(trainingDataSet.get(0).features().size())
//                        .batchSize(1000)
//                        .maximumNumberOfIterations(1000)
//                        .loggingLevel(2)
//                        .sparse(true)
//                        .build();
//        classifier.train(trainingDataSet.subSet(0, 500000));
//        DataSet<PredictedDataInstance<Vector, Double>> testingDataSet = new DataSetInMemory<>();
//        for (LabeledDataInstance<Vector, Double> dataInstance : trainingDataSet.subSet(500000, trainingDataSet.size()))
//            testingDataSet.add(new PredictedDataInstance<>(dataInstance.name(),
//                                                           dataInstance.features(),
//                                                           dataInstance.label(),
//                                                           dataInstance.source(),
//                                                           1));
//        int[] expectedPredictions = new int[testingDataSet.size()];
//        for (int i = 0; i < expectedPredictions.length; i++)
//            expectedPredictions[i] = (int) (double) testingDataSet.get(i).label();
//        DataSet<PredictedDataInstance<Vector, Double>> predictedDataSet = classifier.predictInPlace(testingDataSet);
//        int[] actualPredictions = new int[testingDataSet.size()];
//        for (int i = 0; i < actualPredictions.length; i++)
//            actualPredictions[i] = (int) (double) predictedDataSet.get(i).label();
//        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
//    }

    @Test
    public void testSmallDenseBinaryLogisticRegressionUsingAdaGradAndDataSetWithPredictedDataInstances() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/fisher.binary.txt";
        DataSet<PredictedDataInstance<Vector, Double>> trainingDataSet = Utilities.parseFisherDataFromFile(filename);
        LogisticRegressionAdaGrad classifier =
                new LogisticRegressionAdaGrad.Builder(trainingDataSet.get(0).features().size())
                        .sparse(false)
                        .loggingLevel(1)
                        .build();
        classifier.train(trainingDataSet.subSet(0, 80));
        DataSet<PredictedDataInstance<Vector, Double>> testingDataSet = new DataSetInMemory<>();
        for (PredictedDataInstance<Vector, Double> dataInstance : trainingDataSet.subSet(80, trainingDataSet.size()))
            testingDataSet.add(dataInstance);
        int[] expectedPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < expectedPredictions.length; i++)
            expectedPredictions[i] = (int) (double) testingDataSet.get(i).label();
        DataSet<PredictedDataInstance<Vector, Double>> predictedDataSet = classifier.predictInPlace(testingDataSet);
        int[] actualPredictions = new int[testingDataSet.size()];
        for (int i = 0; i < actualPredictions.length; i++)
            actualPredictions[i] = (int) (double) predictedDataSet.get(i).label();
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

//    @Test
//    public void testLargeSparseBinaryLogisticRegressionUsingAdaGrad() {
//        String filename = "/Users/Anthony/Development/Data Sets/Classification/url.binary.txt";
//        DataSet<PredictedDataInstance<Vector, Double>> trainingDataSet = Utilities.parseURLDataFromFile(filename, true);
//        LogisticRegressionAdaGrad classifier =
//                new LogisticRegressionAdaGrad.Builder(trainingDataSet.get(0).features().size())
//                        .sparse(true)
//                        .maximumNumberOfIterations(100000)
//                        .batchSize(10)
//                        .loggingLevel(2)
//                        .useBiasTerm(false)
//                        .build();
//        classifier.train(trainingDataSet.subSet(0, 20000));
//        DataSet<PredictedDataInstance<Vector, Double>> testingDataSet = new DataSetInMemory<>();
//        for (LabeledDataInstance<Vector, Double> dataInstance : trainingDataSet.subSet(20000, trainingDataSet.size()))
//            testingDataSet.add(new PredictedDataInstance<>(dataInstance.name(),
//                                                           dataInstance.features(),
//                                                           dataInstance.label(),
//                                                           dataInstance.source(),
//                                                           1));
//        int[] expectedPredictions = new int[testingDataSet.size()];
//        for (int i = 0; i < expectedPredictions.length; i++)
//            expectedPredictions[i] = (int) (double) testingDataSet.get(i).label();
//        DataSet<PredictedDataInstance<Vector, Double>> predictedDataSet = classifier.predictInPlace(testingDataSet);
//        int[] actualPredictions = new int[testingDataSet.size()];
//        double accuracy = 0;
//        for (int i = 0; i < actualPredictions.length; i++) {
//            actualPredictions[i] = (int) (double) predictedDataSet.get(i).label();
//            accuracy += actualPredictions[i] == expectedPredictions[i] ? 1 : 0;
//        }
//        accuracy /= actualPredictions.length;
//        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
//    }

    @Test
    public void testDenseBinaryLogisticRegressionPrediction() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        DataSet<PredictedDataInstance<Vector, Double>> trainingDataSet = Utilities.parseCovTypeDataFromFile(filename, false);
        LogisticRegressionSGD classifier =
                new LogisticRegressionSGD.Builder(trainingDataSet.get(0).features().size())
                        .sparse(false)
                        .build();
        classifier.train(trainingDataSet.subSet(0, 500000));
        try {
            filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.dense.model";
            FileOutputStream fout = new FileOutputStream(filename);
            classifier.write(fout, true);
            fout.close();
            FileInputStream fin = new FileInputStream(filename);
            LogisticRegressionPrediction loadedClassifier = LogisticRegressionPrediction.read(fin, true);
            fin.close();
            DataSet<PredictedDataInstance<Vector, Double>> testingDataSet = new DataSetInMemory<>();
            for (LabeledDataInstance<Vector, Double> dataInstance : trainingDataSet.subSet(500000, trainingDataSet.size()))
                testingDataSet.add(new PredictedDataInstance<>(dataInstance.name(),
                                                               dataInstance.features(),
                                                               dataInstance.label(),
                                                               dataInstance.source(),
                                                               1));
            int[] expectedPredictions = new int[testingDataSet.size()];
            for (int i = 0; i < expectedPredictions.length; i++)
                expectedPredictions[i] = (int) (double) testingDataSet.get(i).label();
            DataSet<PredictedDataInstance<Vector, Double>> predictedDataSet = loadedClassifier.predictInPlace(testingDataSet);
            int[] actualPredictions = new int[testingDataSet.size()];
            for (int i = 0; i < actualPredictions.length; i++)
                actualPredictions[i] = (int) (double) predictedDataSet.get(i).label();
            Assert.assertArrayEquals(expectedPredictions, actualPredictions);
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void testSparseBinaryLogisticRegressionPrediction() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        DataSet<PredictedDataInstance<Vector, Double>> trainingDataSet = Utilities.parseCovTypeDataFromFile(filename, true);
        LogisticRegressionSGD classifier =
                new LogisticRegressionSGD.Builder(trainingDataSet.get(0).features().size())
                        .sparse(true)
                        .build();
        classifier.train(trainingDataSet.subSet(0, 500000));
        try {
            filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.sparse.model";
            FileOutputStream fout = new FileOutputStream(filename);
            classifier.write(fout, true);
            fout.close();
            FileInputStream fin = new FileInputStream(filename);
            LogisticRegressionPrediction loadedClassifier = LogisticRegressionPrediction.read(fin, true);
            fin.close();
            DataSet<PredictedDataInstance<Vector, Double>> testingDataSet = new DataSetInMemory<>();
            for (LabeledDataInstance<Vector, Double> dataInstance : trainingDataSet.subSet(500000, trainingDataSet.size()))
                testingDataSet.add(new PredictedDataInstance<>(dataInstance.name(),
                                                               dataInstance.features(),
                                                               dataInstance.label(),
                                                               dataInstance.source(),
                                                               1));
            int[] expectedPredictions = new int[testingDataSet.size()];
            for (int i = 0; i < expectedPredictions.length; i++)
                expectedPredictions[i] = (int) (double) testingDataSet.get(i).label();
            DataSet<PredictedDataInstance<Vector, Double>> predictedDataSet = loadedClassifier.predictInPlace(testingDataSet);
            int[] actualPredictions = new int[testingDataSet.size()];
            for (int i = 0; i < actualPredictions.length; i++)
                actualPredictions[i] = (int) (double) predictedDataSet.get(i).label();
            Assert.assertArrayEquals(expectedPredictions, actualPredictions);
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }
}
