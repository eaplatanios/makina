package org.platanios.learn.classification;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.math.MathUtilities;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class MultiClassLogisticRegressionTest {
    @Test
    public void testLogisticRegressionUsingSGDWithConstantStepSize() {
        String filename = MultiClassLogisticRegressionTest.class.getResource("/FishersIris.csv").getPath();
        TrainingData data = parseLabeledDataFromCSVFile(filename);
        MultiClassLogisticRegression classifier = new MultiClassLogisticRegression.Builder(data)
                .stochastic(true)
//                .batchSize(5)
//                .pointChangeTolerance(1e-2)
//                .maximumNumberOfIterationsWithNoPointChange(10)
//                .stepSize(StochasticSolverStepSize.CONSTANT)
//                .stepSizeParameters(1)
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
            actualPredictions[i] = MathUtilities.computeArgMax(actualPredictionsProbabilities[i]);
        }
        int[] expectedPredictions = new int[] { 1, 0, 2, 1, 0, 1, 2, 0, 1, 0 };
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

//    @Test
//    public void testLogisticRegressionUsingSGDWithScaledStepSize() {
//        String filename = LogisticRegressionTest.class.getResource("/FishersIris.csv").getPath();
//        TrainingData data = parseLabeledDataFromCSVFile(filename);
//        LogisticRegression classifier = new LogisticRegression.Builder(data)
//                .stochastic(true)
//                .batchSize(5)
//                .maximumNumberOfIterationsWithNoPointChange(10)
//                .stepSize(AbstractStochasticIterativeSolver.StepSize.SCALED)
//                .stepSizeParameters(10, 0.75)
//                .build();
//        classifier.train();
//        double[][] actualPredictionsProbabilities = classifier.predict(new double[][] {
//                { 22, 58, 30, 65 },
//                { 3, 14, 35, 51 },
//                { 14, 47, 29, 61 },
//                { 19, 53, 27, 64 },
//                { 2, 16, 34, 48 },
//                { 20, 50, 25, 57 },
//                { 13, 40, 23, 55 },
//                { 2, 17, 34, 54 },
//                { 24, 51, 28, 58 },
//                { 2, 15, 37, 53 }
//        });
//        int[] actualPredictions = new int[actualPredictionsProbabilities.length];
//        for (int i = 0; i < actualPredictions.length; i++) {
//            actualPredictions[i] = Utilities.computeArgMax(actualPredictionsProbabilities[i]);
//        }
//        int[] expectedPredictions = new int[] { 1, 0, 2, 1, 0, 1, 2, 0, 1, 0 };
//        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
//    }

    @Test
    public void testLogisticRegressionUsingBFGS() {
        String filename = MultiClassLogisticRegressionTest.class.getResource("/FishersIris.csv").getPath();
        TrainingData data = parseLabeledDataFromCSVFile(filename);
        MultiClassLogisticRegression classifier = new MultiClassLogisticRegression.Builder(data)
                .stochastic(false)
                .largeScale(false)
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
            actualPredictions[i] = MathUtilities.computeArgMax(actualPredictionsProbabilities[i]);
        }
        int[] expectedPredictions = new int[] { 1, 0, 2, 1, 0, 1, 2, 0, 1, 0 };
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    @Test
    public void testLogisticRegressionUsingLBFGS() {
        String filename = MultiClassLogisticRegressionTest.class.getResource("/FishersIris.csv").getPath();
        TrainingData data = parseLabeledDataFromCSVFile(filename);
        MultiClassLogisticRegression classifier = new MultiClassLogisticRegression.Builder(data)
                .stochastic(false)
                .largeScale(true)
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
            actualPredictions[i] = MathUtilities.computeArgMax(actualPredictionsProbabilities[i]);
        }
        int[] expectedPredictions = new int[] { 1, 0, 2, 1, 0, 1, 2, 0, 1, 0 };
        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
    }

    public static TrainingData parseLabeledDataFromCSVFile(String filename) {
        String separator = ",";

        int numberOfFeatures;
        List<Integer> labels = new ArrayList<>();
        List<Vector> data = new ArrayList<>();

        BufferedReader br = null;
        String line;
        String[] classifiersNames = null;

        try {
            br = new BufferedReader(new FileReader(filename));
            line = br.readLine();
            classifiersNames = line.split(separator);
            numberOfFeatures = classifiersNames.length - 1;
            while ((line = br.readLine()) != null) {
                String[] outputs = line.split(separator);
                labels.add(Integer.parseInt(outputs[0]));
                double[] dataSample = new double[numberOfFeatures];
                for (int i = 0; i < numberOfFeatures; i++) {
                    dataSample[i] = Double.parseDouble(outputs[i+1]);
                }
                data.add(Vectors.dense(dataSample));
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        if (data.size() != labels.size()) {
            throw new IllegalArgumentException("The number of data labels and data samples does not match."); // TODO: Maybe the IllegalArgumentException is not the most appropriate here.
        }

        List<TrainingData.Entry> trainingData = new ArrayList<>();
        for (int i = 0; i < data.size(); i++) {
            trainingData.add(new TrainingData.Entry(data.get(i), labels.get(i)));
        }

        return new TrainingData(trainingData);
    }
}
