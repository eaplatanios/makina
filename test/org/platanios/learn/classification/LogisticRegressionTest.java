package org.platanios.learn.classification;

import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LogisticRegressionTest {
    @Test
    public void testLogisticRegression() {
//        String filename = LogisticRegressionTest.class.getResource("resources/FisherIris.csv").getPath();
        String filename = "/Users/Anthony/Development/GitHub/learn/test/org/platanios/learn/classification/resources/FishersIris.csv";
        TrainingData data = DataPreprocessing.parseLabeledDataFromCSVFile(filename);
        LogisticRegression classifier = new LogisticRegression(data.getData(), data.getLabels());
        classifier.train();
        double[] actualPredictionsProbabilities = classifier.predict(new double[][] {
                { 3, 14, 35, 51 },
                { 3, 13, 35, 50 },
                { 2, 16, 34, 48 },
                { 2, 17, 34, 54 },
                { 2, 15, 37, 53 },
                { 19, 61, 28, 74 },
                { 22, 58, 30, 65 },
                { 19, 53, 27, 64 },
                { 20, 50, 25, 57 },
                { 24, 51, 28, 58 }
        });
        int[] actualPredictions = new int[actualPredictionsProbabilities.length];
        for (int i = 0; i < actualPredictions.length; i++) {
            actualPredictions[i] = actualPredictionsProbabilities[i] >= 0.5 ? 1 : 0;
        }
        int[] expectedPredictions = new int[] { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 };
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }
}
