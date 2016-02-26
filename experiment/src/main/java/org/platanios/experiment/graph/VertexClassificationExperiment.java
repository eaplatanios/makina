package org.platanios.experiment.graph;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.evaluation.PrecisionRecall;
import org.platanios.learn.graph.Graph;
import org.platanios.learn.graph.Vertex;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.math.statistics.StatisticsUtilities;
import org.platanios.learn.neural.graph.FeatureVectorFunctionType;
import org.platanios.learn.neural.graph.GraphRecursiveNeuralNetwork;
import org.platanios.learn.neural.graph.VertexClassificationRecursiveNeuralNetwork;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public class VertexClassificationExperiment {
    private final Logger logger = LogManager.getFormatterLogger("Vertex Classification Experiment");
    private final Map<Integer, Vertex<GraphRecursiveNeuralNetwork.VertexContentType, Void>> vertexIndexesMap = new HashMap<>();

    private final String dataSetName;
    private final Graph<GraphRecursiveNeuralNetwork.VertexContentType, Void> graph;
    private final FeatureVectorFunctionType featureVectorFunctionType;
    private final Map<Integer, Integer> trueLabels;
    private final String evaluationResultsFolderPath;

    public VertexClassificationExperiment(DataSets.LabeledDataSet dataSet,
                                          FeatureVectorFunctionType featureVectorFunctionType,
                                          String evaluationResultsFolderPath) {
        dataSetName = dataSet.getName();
        logger.info("Number of vertices: " + dataSet.getVertexIndices().size());
        logger.info("Number of edges: " + dataSet.getEdges().size());
        graph = new Graph<>();
        for (int vertexIndex : dataSet.getVertexIndices())
            vertexIndexesMap.put(vertexIndex, new Vertex<>(new GraphRecursiveNeuralNetwork.VertexContentType(vertexIndex, 0, null, null)));
        for (DataSets.Edge edge : dataSet.getEdges())
            graph.addEdge(vertexIndexesMap.get(edge.getSourceVertexIndex()), vertexIndexesMap.get(edge.getDestinationVertexIndex()));
        logger.info("Finished generating graphs for the " + dataSetName + " data set.");
        trueLabels = new HashMap<>(dataSet.getVertexLabels());
        this.featureVectorFunctionType = featureVectorFunctionType;
        this.evaluationResultsFolderPath = evaluationResultsFolderPath;
    }

    public void runRNNExperiments(int[] featureVectorsSizes, int[] maximumNumberOfStepsToTry, int numberOfFolds) {
        EvaluationResults[][] results = new EvaluationResults[featureVectorsSizes.length][maximumNumberOfStepsToTry.length];
        for (int i = 0; i < featureVectorsSizes.length; i++)
            for (int j = 0; j < maximumNumberOfStepsToTry.length; j++) {
                EvaluationResults[] currentSettingResults = new EvaluationResults[numberOfFolds];
                final List<Map.Entry<Integer, Integer>> entries = new ArrayList<>(trueLabels.entrySet());
                Collections.shuffle(entries);
                int foldSize = Math.floorDiv(entries.size(), numberOfFolds);
                for (int foldNumber = 0; foldNumber < numberOfFolds; foldNumber++) {
                    Map<Integer, Vector> trainingData = new HashMap<>();
                    for (Map.Entry<Integer, Integer> entry : entries.subList(0, foldNumber * foldSize))
                        trainingData.put(entry.getKey(), Vectors.dense((double) entry.getValue()));
                    for (Map.Entry<Integer, Integer> entry : entries.subList((foldNumber + 1) * foldSize, entries.size()))
                        trainingData.put(entry.getKey(), Vectors.dense((double) entry.getValue()));
                    currentSettingResults[foldNumber] = runRNNExperiment(featureVectorsSizes[i],
                                                                         maximumNumberOfStepsToTry[j],
                                                                         trainingData);
                }
                results[i][j] = EvaluationResults.averageResults(currentSettingResults);
                appendEvaluationResultsToFile(featureVectorsSizes[i],
                                              maximumNumberOfStepsToTry[j],
                                              results[i][j],
                                              evaluationResultsFolderPath);
            }
        logger.info("Results for the " + dataSetName + " data set:");
        for (int i = 0; i < featureVectorsSizes.length; i++)
            for (int j = 0; j < maximumNumberOfStepsToTry.length; j++)
                logger.info("\tF = %d, K = %d:\t { Train Acc.: %20s | Test Acc.: %20s | Overall Acc.: %20s }",
                            featureVectorsSizes[i],
                            maximumNumberOfStepsToTry[j],
                            results[i][j].trainAccuracy,
                            results[i][j].testAccuracy,
                            results[i][j].overallAccuracy);
    }

    public EvaluationResults runRNNExperiment(int featureVectorsSize, int maximumNumberOfSteps, Map<Integer, Vector> trainingData) {
        VertexClassificationRecursiveNeuralNetwork<Void> graphRNNAlgorithm =
                new VertexClassificationRecursiveNeuralNetwork<>(featureVectorsSize, 1, maximumNumberOfSteps, graph, trainingData, featureVectorFunctionType);
        if (!graphRNNAlgorithm.checkDerivative(1e-5))
            logger.warn("The derivatives of the RNN objective function provided are not the same as those obtained " +
                                "by the method of finite differences.");
        logger.info("Training RNN for the " + dataSetName + " data set.");
        graphRNNAlgorithm.trainNetwork();
        graphRNNAlgorithm.performForwardPass();
        logger.info("Finished training RNN for the " + dataSetName + " data set.");
        logger.info("Storing RNN results for the " + dataSetName + " data set.");
        List<PredictedDataInstance<Vector, Integer>> trainPredictions = new ArrayList<>();
        List<PredictedDataInstance<Vector, Integer>> testPredictions = new ArrayList<>();
        List<PredictedDataInstance<Vector, Integer>> allPredictions = new ArrayList<>();
        graph.getVertices()
                .stream()
                .filter(vertex -> trueLabels.containsKey(vertex.getContent().getId()))
                .forEach(vertex -> {
                    PredictedDataInstance<Vector, Integer> instance = new PredictedDataInstance<>(
                            null,
                            1,
                            vertex.getContent().getId(),
                            graphRNNAlgorithm.getOutputForVertex(vertex).get(0)
                    );
                    if (trainingData.containsKey(vertex.getContent().getId()))
                        trainPredictions.add(instance);
                    else
                        testPredictions.add(instance);
                    allPredictions.add(instance);
                });
        logger.info("Finished storing RNN results for the " + dataSetName + " data set.");
        EvaluationResults results = evaluateResults(trainPredictions, testPredictions, allPredictions);
        graphRNNAlgorithm.resetGraph();
        return results;
    }

    private EvaluationResults evaluateResults(List<PredictedDataInstance<Vector, Integer>> trainPredictions,
                                              List<PredictedDataInstance<Vector, Integer>> testPredictions,
                                              List<PredictedDataInstance<Vector, Integer>> allPredictions) {
        PrecisionRecall<Vector, Integer> precisionRecall = new PrecisionRecall<>(1000);
        precisionRecall.addResult("Train", trainPredictions, instance -> (instance.probability() >= 0.5 ? 1 : 0) == trueLabels.get(instance.source()));
        precisionRecall.addResult("Test", testPredictions, instance -> (instance.probability() >= 0.5 ? 1 : 0) == trueLabels.get(instance.source()));
        precisionRecall.addResult("All", allPredictions, instance -> (instance.probability() >= 0.5 ? 1 : 0) == trueLabels.get(instance.source()));
        double trainAccuracy = precisionRecall.getAreaUnderCurve("Train");
        double testAccuracy = precisionRecall.getAreaUnderCurve("Test");
        double overallAccuracy = precisionRecall.getAreaUnderCurve("All");
        return new EvaluationResults(trainAccuracy, testAccuracy, overallAccuracy);
    }

    public static class EvaluationResults {
        private final double trainAccuracy;
        private final double testAccuracy;
        private final double overallAccuracy;
        private final double trainAccuracyStandardDeviation;
        private final double testAccuracyStandardDeviation;
        private final double overallAccuracyStandardDeviation;

        public EvaluationResults(double trainAccuracy, double testAccuracy, double overallAccuracy) {
            this(trainAccuracy, testAccuracy, overallAccuracy, 0.0, 0.0, 0.0);
        }

        public EvaluationResults(double trainAccuracy,
                                 double testAccuracy,
                                 double overallAccuracy,
                                 double trainAccuracyStandardDeviation,
                                 double testAccuracyStandardDeviation,
                                 double overallAccuracyStandardDeviation) {
            this.trainAccuracy = trainAccuracy;
            this.testAccuracy = testAccuracy;
            this.overallAccuracy = overallAccuracy;
            this.trainAccuracyStandardDeviation = trainAccuracyStandardDeviation;
            this.testAccuracyStandardDeviation = testAccuracyStandardDeviation;
            this.overallAccuracyStandardDeviation = overallAccuracyStandardDeviation;
        }

        public double getTrainAccuracy() {
            return trainAccuracy;
        }

        public double getTestAccuracy() {
            return testAccuracy;
        }

        public double getOverallAccuracy() {
            return overallAccuracy;
        }

        public double getTrainAccuracyStandardDeviation() {
            return trainAccuracyStandardDeviation;
        }

        public double getTestAccuracyStandardDeviation() {
            return testAccuracyStandardDeviation;
        }

        public double getOverallAccuracyStandardDeviation() {
            return overallAccuracyStandardDeviation;
        }

        public static EvaluationResults averageResults(EvaluationResults... results) {
            double[] trainAccuracies = new double[results.length];
            double[] testAccuracies = new double[results.length];
            double[] overallAccuracies = new double[results.length];
            for (int resultIndex = 0; resultIndex < results.length; resultIndex++) {
                trainAccuracies[resultIndex] = results[resultIndex].getTrainAccuracy();
                testAccuracies[resultIndex] = results[resultIndex].getTestAccuracy();
                overallAccuracies[resultIndex] = results[resultIndex].getOverallAccuracy();
            }
            return new EvaluationResults(StatisticsUtilities.mean(trainAccuracies),
                                         StatisticsUtilities.mean(testAccuracies),
                                         StatisticsUtilities.mean(overallAccuracies),
                                         StatisticsUtilities.standardDeviation(trainAccuracies),
                                         StatisticsUtilities.standardDeviation(testAccuracies),
                                         StatisticsUtilities.standardDeviation(overallAccuracies));
        }
    }

    private void appendEvaluationResultsToFile(int featureVectorsSize,
                                               int maximumNumberOfSteps,
                                               EvaluationResults results,
                                               String folderPath) {
        try {
            String resultsLine = featureVectorsSize + "\t" + maximumNumberOfSteps + "\t" +
                    results.getTrainAccuracy() + "|" + results.getTrainAccuracyStandardDeviation() + "\t" +
                    results.getTestAccuracy() + "|" + results.getTestAccuracyStandardDeviation() + "\t" +
                    results.getOverallAccuracy() + "|" + results.getOverallAccuracyStandardDeviation() + "\n";
            Files.write(Paths.get(folderPath + "/results_" + normalizeDataSetName(dataSetName) + "_" + featureVectorFunctionType.name().toLowerCase() + ".txt"),
                        resultsLine.getBytes(),
                        StandardOpenOption.CREATE,
                        StandardOpenOption.APPEND);
        } catch (IOException e) {
            logger.error("Failed to append evaluation results to the provided file.", e);
        }
    }

    private String normalizeDataSetName(String dataSetName) {
        return dataSetName.replace(" ", "_").toLowerCase();
    }

    public static void main(String[] args) {
        DataSets.LabeledDataSet dataSet = DataSets.loadLabeledDataSet(args[4]);
        VertexClassificationExperiment experiment = new VertexClassificationExperiment(dataSet, FeatureVectorFunctionType.valueOf(args[0]), args[5]);
        experiment.runRNNExperiments(intArrayFromString(args[1]), intArrayFromString(args[2]), Integer.parseInt(args[3]));
    }

    private static int[] intArrayFromString(String string) {
        String[] stringParts = string.split(",");
        int[] array = new int[stringParts.length];
        for (int partIndex = 0; partIndex < stringParts.length; partIndex++)
            array[partIndex] = Integer.parseInt(stringParts[partIndex]);
        return array;
    }
}
