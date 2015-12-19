package org.platanios.learn.classification;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.DataSetInMemory;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class EvaluationDataSetTrainingTest {
    @Test
    public void testGridSearchFisherDataSet() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/fisher.binary.txt";
        DataSet<PredictedDataInstance<Vector, Double>> trainingDataSet = Utilities.parseFisherDataFromFile(filename);
        LogisticRegressionAdaGrad.Builder classifierBuilder =
                new LogisticRegressionAdaGrad.Builder(trainingDataSet.get(0).features().size())
                        .sparse(false)
                        .maximumNumberOfIterations(1000)
                        .maximumNumberOfIterationsWithNoPointChange(10)
                        .loggingLevel(0);
        EvaluationDataSetTraining<Vector, Double> training =
                new EvaluationDataSetTraining.Builder<>(classifierBuilder,
                                                        trainingDataSet.subSet(0, 60),
                                                        trainingDataSet.subSet(60, 80))
                        .addAllowedParameterValues("sampleWithReplacement", true, false)
                        .addAllowedParameterValues("batchSize", 1, 10, 20)
                        .addAllowedParameterValues("useBiasTerm", true, false )
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

    @Test
    public void testGridSearchCovTypeDataSet() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        DataSet<PredictedDataInstance<Vector, Double>> trainingDataSet = Utilities.parseCovTypeDataFromFile(filename, true);
        LogisticRegressionAdaGrad.Builder classifierBuilder =
                new LogisticRegressionAdaGrad.Builder(trainingDataSet.get(0).features().size())
                        .maximumNumberOfIterations(1000)
                        .loggingLevel(0)
                        .sparse(true);
        EvaluationDataSetTraining<Vector, Double> training =
                new EvaluationDataSetTraining.Builder<>(classifierBuilder,
                                                        trainingDataSet.subSet(0, 200000),
                                                        trainingDataSet.subSet(200000, 500000))
                        .addAllowedParameterValues("sampleWithReplacement", true, false)
                        .addAllowedParameterValues("batchSize", 1, 10, 20)
                        .addAllowedParameterValues("useBiasTerm", true, false )
                        .addAllowedParameterValues("l1RegularizationWeight", 0.1, 1.0, 10.0)
                        .addAllowedParameterValues("l2RegularizationWeight", 0.1, 1.0, 10.0)
                        .build();
        LogisticRegressionAdaGrad classifier = (LogisticRegressionAdaGrad) training.train().getClassifier();
        DataSet<PredictedDataInstance<Vector, Double>> testingDataSet = new DataSetInMemory<>();
        for (PredictedDataInstance<Vector, Double> dataInstance : trainingDataSet.subSet(500000, trainingDataSet.size()))
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
    public void testGridSearchURLDataSet() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/url.binary.txt";
        DataSet<PredictedDataInstance<Vector, Double>> trainingDataSet = Utilities.parseURLDataFromFile(filename, true);
        LogisticRegressionAdaGrad.Builder classifierBuilder =
                new LogisticRegressionAdaGrad.Builder(trainingDataSet.get(0).features().size())
                        .sparse(true)
                        .maximumNumberOfIterations(1000)
                        .maximumNumberOfIterationsWithNoPointChange(10)
                        .loggingLevel(0);
        EvaluationDataSetTraining<Vector, Double> training =
                new EvaluationDataSetTraining.Builder<>(classifierBuilder,
                                                        trainingDataSet.subSet(0, 10000),
                                                        trainingDataSet.subSet(10000, 20000))
                        .addAllowedParameterValues("sampleWithReplacement", true, false)
                        .addAllowedParameterValues("batchSize", 1, 10)
                        .addAllowedParameterValues("useBiasTerm", true, false )
                        .addAllowedParameterValues("l1RegularizationWeight", 0.1, 1.0, 10.0)
                        .addAllowedParameterValues("l2RegularizationWeight", 0.1, 1.0, 10.0)
                        .build();
        LogisticRegressionAdaGrad classifier = (LogisticRegressionAdaGrad) training.train().getClassifier();
        DataSet<PredictedDataInstance<Vector, Double>> testingDataSet = new DataSetInMemory<>();
        for (PredictedDataInstance<Vector, Double> dataInstance : trainingDataSet.subSet(20000, trainingDataSet.size()))
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
}
