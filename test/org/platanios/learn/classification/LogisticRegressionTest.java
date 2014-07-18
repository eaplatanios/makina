package org.platanios.learn.classification;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.math.Utilities;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LogisticRegressionTest {
    @Test
    public void testLogisticRegression() {
        String filename = LogisticRegressionTest.class.getResource("resources/FishersIris.csv").getPath();
        TrainingData data = DataPreprocessing.parseLabeledDataFromCSVFile(filename);
        LogisticRegression classifier = new LogisticRegression.Builder(data)
                .stochastic(true)
                .batchSize(1)
                .tau(1000)
                .kappa(1)
                .pointChangeTolerance(1e-5)
                .build();
        classifier.train();
        double[][] actualPredictionsProbabilities = classifier.predict(new double[][] {
                { 22, 58, 30, 65 },
                { 3, 14, 35, 51 },
                { 14, 47, 29, 61 },
                { 19, 53, 27, 64 },
                { 2, 16, 34, 48 },
                { 20, 50, 25, 57 },
                { 13, 40, 23, 55 },
                { 2, 17, 34, 54 },
                { 24, 51, 28, 58 },
                { 2, 15, 37, 53 }
        });
        int[] actualPredictions = new int[actualPredictionsProbabilities.length];
        for (int i = 0; i < actualPredictions.length; i++) {
            actualPredictions[i] = Utilities.computeArgMax(actualPredictionsProbabilities[i]);
        }
        int[] expectedPredictions = new int[] { 1, 0, 2, 1, 0, 1, 2, 0, 1, 0 };
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }
}
