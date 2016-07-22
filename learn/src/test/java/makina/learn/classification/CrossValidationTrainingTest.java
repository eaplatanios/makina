package makina.learn.classification;

import org.junit.Assert;
import org.junit.Test;
import makina.learn.data.DataSet;
import makina.learn.data.DataSetInMemory;
import makina.learn.data.PredictedDataInstance;
import makina.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class CrossValidationTrainingTest {
    @Test
    public void testGridSearchFisherDataSet() {
        String filename = "/Users/Anthony/Development/Data/Classification/fisher.binary.txt";
        DataSet<PredictedDataInstance<Vector, Double>> trainingDataSet = Utilities.parseFisherDataFromFile(filename);
        LogisticRegressionAdaGrad.Builder classifierBuilder =
                new LogisticRegressionAdaGrad.Builder(trainingDataSet.get(0).features().size())
                        .sparse(false)
                        .maximumNumberOfIterations(1000)
                        .maximumNumberOfIterationsWithNoPointChange(10)
                        .loggingLevel(0);
        CrossValidationTraining<Vector, Double> training =
                new CrossValidationTraining.Builder<>(classifierBuilder, trainingDataSet.subSet(0, 80))
                        .numberOfFolds(10)
                        .addAllowedParameterValues("sampleWithReplacement", true, false)
                        .addAllowedParameterValues("batchSize", 1, 10, 20)
                        .addAllowedParameterValues("useBiasTerm", true, false)
                        .addAllowedParameterValues("l1RegularizationWeight", 0.1, 1.0, 10.0)
                        .addAllowedParameterValues("l2RegularizationWeight", 0.1, 1.0, 10.0)
                        .build();
        LogisticRegressionAdaGrad classifier = (LogisticRegressionAdaGrad) training.train().getClassifier();
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
//    public void testGridSearchCovTypeDataSet() {
//        String filename = "/Users/Anthony/Development/Data/Classification/covtype.binary.scale.txt";
//        DataSet<PredictedDataInstance<Vector, Double>> trainingDataSet = Utilities.parseCovTypeDataFromFile(filename, true);
//        LogisticRegressionAdaGrad.Builder classifierBuilder =
//                new LogisticRegressionAdaGrad.Builder(trainingDataSet.get(0).features().size())
//                        .maximumNumberOfIterations(1000)
//                        .loggingLevel(0)
//                        .sparse(true);
//        CrossValidationTraining<Vector, Double> training =
//                new CrossValidationTraining.Builder<>(classifierBuilder, trainingDataSet.subSet(0, 500000))
//                        .numberOfFolds(10)
//                        .addAllowedParameterValues("sampleWithReplacement", true, false)
//                        .addAllowedParameterValues("batchSize", 1, 10, 20)
//                        .addAllowedParameterValues("useBiasTerm", true, false)
//                        .addAllowedParameterValues("l1RegularizationWeight", 0.1, 1.0, 10.0)
//                        .addAllowedParameterValues("l2RegularizationWeight", 0.1, 1.0, 10.0)
//                        .build();
//        LogisticRegressionAdaGrad classifier = (LogisticRegressionAdaGrad) training.train().getClassifier();
//        DataSet<PredictedDataInstance<Vector, Double>> testingDataSet = new DataSetInMemory<>();
//        for (PredictedDataInstance<Vector, Double> dataInstance : trainingDataSet.subSet(500000, trainingDataSet.size()))
//            testingDataSet.add(dataInstance);
//        int[] expectedPredictions = new int[testingDataSet.size()];
//        for (int i = 0; i < expectedPredictions.length; i++)
//            expectedPredictions[i] = (int) (double) testingDataSet.get(i).label();
//        DataSet<PredictedDataInstance<Vector, Double>> predictedDataSet = classifier.predictInPlace(testingDataSet);
//        int[] actualPredictions = new int[testingDataSet.size()];
//        for (int i = 0; i < actualPredictions.length; i++)
//            actualPredictions[i] = (int) (double) predictedDataSet.get(i).label();
//        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
//    }
//
//    @Test
//    public void testGridSearchURLDataSet() {
//        String filename = "/Users/Anthony/Development/Data/Classification/url.binary.txt";
//        DataSet<PredictedDataInstance<Vector, Double>> trainingDataSet = Utilities.parseURLDataFromFile(filename, true);
//        LogisticRegressionAdaGrad.Builder classifierBuilder =
//                new LogisticRegressionAdaGrad.Builder(trainingDataSet.get(0).features().size())
//                        .sparse(true)
//                        .maximumNumberOfIterations(1000)
//                        .maximumNumberOfIterationsWithNoPointChange(10)
//                        .loggingLevel(0);
//        CrossValidationTraining<Vector, Double> training =
//                new CrossValidationTraining.Builder<>(classifierBuilder, trainingDataSet.subSet(0, 20000))
//                        .numberOfFolds(10)
//                        .addAllowedParameterValues("sampleWithReplacement", true, false)
//                        .addAllowedParameterValues("batchSize", 1, 10)
//                        .addAllowedParameterValues("useBiasTerm", true, false)
//                        .addAllowedParameterValues("l1RegularizationWeight", 0.1, 1.0, 10.0)
//                        .addAllowedParameterValues("l2RegularizationWeight", 0.1, 1.0, 10.0)
//                        .build();
//        LogisticRegressionAdaGrad classifier = (LogisticRegressionAdaGrad) training.train().getClassifier();
//        DataSet<PredictedDataInstance<Vector, Double>> testingDataSet = new DataSetInMemory<>();
//        for (PredictedDataInstance<Vector, Double> dataInstance : trainingDataSet.subSet(20000, trainingDataSet.size()))
//            testingDataSet.add(dataInstance);
//        int[] expectedPredictions = new int[testingDataSet.size()];
//        for (int i = 0; i < expectedPredictions.length; i++)
//            expectedPredictions[i] = (int) (double) testingDataSet.get(i).label();
//        DataSet<PredictedDataInstance<Vector, Double>> predictedDataSet = classifier.predictInPlace(testingDataSet);
//        int[] actualPredictions = new int[testingDataSet.size()];
//        for (int i = 0; i < actualPredictions.length; i++)
//            actualPredictions[i] = (int) (double) predictedDataSet.get(i).label();
//        Assert.assertArrayEquals(expectedPredictions, actualPredictions);
//    }
}
