package org.platanios.learn.classification;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.math.matrix.Vector;

import java.util.Arrays;

/**
 * @author Emmanouil Antonios Platanios
 */
public class BinaryLogisticRegressionTest {
    @Test
    public void testDenseBinaryLogisticRegressionUsingSGD() {
        String filename = LogisticRegressionTest.class.getResource("/covtype.binary.scale.txt").getPath();
        DataInstance<Vector, Integer>[] data = DataPreprocessing.parseLabeledDataFromLIBSVMFile(filename, false);
        BinaryLogisticRegressionSGD classifier =
                new BinaryLogisticRegressionSGD.Builder(Arrays.copyOfRange(data, 0, 500000))
                .sparse(false)
                .build();
        classifier.train();
        double[] actualPredictionsProbabilities = classifier.predict(Arrays.copyOfRange(data, 500000, data.length));
        int[] actualPredictions = new int[actualPredictionsProbabilities.length];
        int[] expectedPredictions = new int[actualPredictionsProbabilities.length];
        for (int i = 0; i < actualPredictions.length; i++) {
            actualPredictions[i] = actualPredictionsProbabilities[i] >= 0.5 ? 1 : 0;
            expectedPredictions[i] = data[500000 + i].getLabel();
        }
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testSparseBinaryLogisticRegressionUsingSGD() {
        String filename = LogisticRegressionTest.class.getResource("/covtype.binary.scale.txt").getPath();
        DataInstance<Vector, Integer>[] data = DataPreprocessing.parseLabeledDataFromLIBSVMFile(filename, true);
        BinaryLogisticRegressionSGD classifier =
                new BinaryLogisticRegressionSGD.Builder(Arrays.copyOfRange(data, 0, 500000))
                .sparse(true)
                .build();
        classifier.train();
        double[] actualPredictionsProbabilities = classifier.predict(Arrays.copyOfRange(data, 500000, data.length));
        int[] actualPredictions = new int[actualPredictionsProbabilities.length];
        int[] expectedPredictions = new int[actualPredictionsProbabilities.length];
        for (int i = 0; i < actualPredictions.length; i++) {
            actualPredictions[i] = actualPredictionsProbabilities[i] >= 0.5 ? 1 : 0;
            expectedPredictions[i] = data[500000 + i].getLabel();
        }
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testSmallDenseBinaryLogisticRegressionUsingSGD() {
        String filename = LogisticRegressionTest.class.getResource("/fisher.binary.txt").getPath();
        DataInstance<Vector, Integer>[] data = DataPreprocessing.parseBinaryLabeledDataFromCSVFile(filename);
        BinaryLogisticRegressionSGD classifier = new BinaryLogisticRegressionSGD.Builder(Arrays.copyOfRange(data, 0, 80))
                .sparse(false)
                .build();
        classifier.train();
        double[] actualPredictionsProbabilities = classifier.predict(Arrays.copyOfRange(data, 80, data.length));
        int[] actualPredictions = new int[actualPredictionsProbabilities.length];
        int[] expectedPredictions = new int[actualPredictionsProbabilities.length];
        for (int i = 0; i < actualPredictions.length; i++) {
            actualPredictions[i] = actualPredictionsProbabilities[i] >= 0.5 ? 1 : 0;
            expectedPredictions[i] = data[80 + i].getLabel();
        }
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testDenseBinaryLogisticRegressionUsingAdaGrad() {
        String filename = LogisticRegressionTest.class.getResource("/covtype.binary.scale.txt").getPath();
        DataInstance<Vector, Integer>[] data = DataPreprocessing.parseLabeledDataFromLIBSVMFile(filename, false);
        BinaryLogisticRegressionAdaGrad classifier =
                new BinaryLogisticRegressionAdaGrad.Builder(Arrays.copyOfRange(data, 0, 500000))
                .sparse(false)
                .batchSize(10000)
                .build();
        classifier.train();
        double[] actualPredictionsProbabilities = classifier.predict(Arrays.copyOfRange(data, 500000, data.length));
        int[] actualPredictions = new int[actualPredictionsProbabilities.length];
        int[] expectedPredictions = new int[actualPredictionsProbabilities.length];
        for (int i = 0; i < actualPredictions.length; i++) {
            actualPredictions[i] = actualPredictionsProbabilities[i] >= 0.5 ? 1 : 0;
            expectedPredictions[i] = data[500000 + i].getLabel();
        }
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testSparseBinaryLogisticRegressionUsingAdaGrad() {
        String filename = LogisticRegressionTest.class.getResource("/covtype.binary.scale.txt").getPath();
        DataInstance<Vector, Integer>[] data = DataPreprocessing.parseLabeledDataFromLIBSVMFile(filename, true);
        BinaryLogisticRegressionAdaGrad classifier =
                new BinaryLogisticRegressionAdaGrad.Builder(Arrays.copyOfRange(data, 0, 500000))
                .sparse(true)
                .build();
        classifier.train();
        double[] actualPredictionsProbabilities = classifier.predict(Arrays.copyOfRange(data, 500000, data.length));
        int[] actualPredictions = new int[actualPredictionsProbabilities.length];
        int[] expectedPredictions = new int[actualPredictionsProbabilities.length];
        for (int i = 0; i < actualPredictions.length; i++) {
            actualPredictions[i] = actualPredictionsProbabilities[i] >= 0.5 ? 1 : 0;
            expectedPredictions[i] = data[500000 + i].getLabel();
        }
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testSmallDenseBinaryLogisticRegressionUsingAdaGrad() {
        String filename = LogisticRegressionTest.class.getResource("/fisher.binary.txt").getPath();
        DataInstance<Vector, Integer>[] data = DataPreprocessing.parseBinaryLabeledDataFromCSVFile(filename);
        BinaryLogisticRegressionAdaGrad classifier =
                new BinaryLogisticRegressionAdaGrad.Builder(Arrays.copyOfRange(data, 0, 80))
                .sparse(false)
                .build();
        classifier.train();
        double[] actualPredictionsProbabilities = classifier.predict(Arrays.copyOfRange(data, 80, data.length));
        int[] actualPredictions = new int[actualPredictionsProbabilities.length];
        int[] expectedPredictions = new int[actualPredictionsProbabilities.length];
        for (int i = 0; i < actualPredictions.length; i++) {
            actualPredictions[i] = actualPredictionsProbabilities[i] >= 0.5 ? 1 : 0;
            expectedPredictions[i] = data[80 + i].getLabel();
        }
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }
}
