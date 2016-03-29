package org.platanios.experiment.graph;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.graph.Graph;
import org.platanios.learn.graph.HITSAlgorithm;
import org.platanios.learn.graph.PageRankAlgorithm;
import org.platanios.learn.graph.Vertex;
import org.platanios.math.matrix.Vector;
import org.platanios.math.matrix.Vectors;
import org.platanios.learn.neural.graph.FeatureVectorFunctionType;
import org.platanios.learn.neural.graph.GraphRecursiveNeuralNetwork;
import org.platanios.learn.neural.graph.VertexRankingRecursiveNeuralNetwork;
import org.platanios.utilities.CollectionUtilities;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public class VertexRankingExperiment2 {
    private final Logger logger = LogManager.getFormatterLogger("Vertex Ranking Experiment");
    private final Map<Integer, Vertex<PageRankAlgorithm.VertexContentType, Void>> trainingPRVertexIndexesMap = new HashMap<>();
    private final Map<Integer, Vertex<HITSAlgorithm.VertexContentType, Void>> trainingHITSVertexIndexesMap = new HashMap<>();
    private final Map<Integer, Vertex<PageRankAlgorithm.VertexContentType, Void>> testingPRVertexIndexesMap = new HashMap<>();
    private final Map<Integer, Vertex<HITSAlgorithm.VertexContentType, Void>> testingHITSVertexIndexesMap = new HashMap<>();
    private final Map<Integer, Vertex<GraphRecursiveNeuralNetwork.VertexContentType, Void>> trainingRNNVertexIndexesMap = new HashMap<>();
    private final Map<Integer, Vertex<GraphRecursiveNeuralNetwork.VertexContentType, Void>> testingRNNVertexIndexesMap = new HashMap<>();
    private final Map<Integer, Vector> trueTrainingScores = new TreeMap<>();
    private final Map<Integer, Vector> trueTestingScores = new TreeMap<>();

    private final String trainingDataSetName;
    private final String testingDataSetName;
    private final RankingAlgorithm rankingAlgorithm;
    private final Graph<PageRankAlgorithm.VertexContentType, Void> trainingPRGraph;
    private final Graph<HITSAlgorithm.VertexContentType, Void> trainingHITSGraph;
    private final Graph<PageRankAlgorithm.VertexContentType, Void> testingPRGraph;
    private final Graph<HITSAlgorithm.VertexContentType, Void> testingHITSGraph;
    private final Graph<GraphRecursiveNeuralNetwork.VertexContentType, Void> trainingRNNGraph;
    private final Graph<GraphRecursiveNeuralNetwork.VertexContentType, Void> testingRNNGraph;
    private final FeatureVectorFunctionType featureVectorFunctionType;
    private final String evaluationResultsFolderPath;

    public VertexRankingExperiment2(DataSets.DataSet trainingDataSet,
                                    DataSets.DataSet testingDataSet,
                                    RankingAlgorithm rankingAlgorithm,
                                    FeatureVectorFunctionType featureVectorFunctionType,
                                    String evaluationResultsFolderPath) {
        trainingDataSetName = trainingDataSet.getName();
        testingDataSetName = testingDataSet.getName();
        this.rankingAlgorithm = rankingAlgorithm;
        logger.info("Number of training data set vertices: " + trainingDataSet.getVertexIndices().size());
        logger.info("Number of training data set edges: " + trainingDataSet.getEdges().size());
        logger.info("Number of testing data set vertices: " + testingDataSet.getVertexIndices().size());
        logger.info("Number of testing data set edges: " + testingDataSet.getEdges().size());
        switch (rankingAlgorithm) {
            case PAGERANK:
                trainingPRGraph = new Graph<>();
                trainingHITSGraph = null;
                testingPRGraph = new Graph<>();
                testingHITSGraph = null;
                break;
            case HITS:
                trainingPRGraph = null;
                trainingHITSGraph = new Graph<>();
                testingPRGraph = null;
                testingHITSGraph = new Graph<>();
                break;
            default:
                trainingPRGraph = null;
                trainingHITSGraph = null;
                testingPRGraph = null;
                testingHITSGraph = null;
                break;
        }
        trainingRNNGraph = new Graph<>();
        testingRNNGraph = new Graph<>();
        for (int vertexIndex : trainingDataSet.getVertexIndices()) {
            switch (rankingAlgorithm) {
                case PAGERANK:
                    trainingPRVertexIndexesMap.put(vertexIndex, new Vertex<>(new PageRankAlgorithm.VertexContentType(vertexIndex, 0.0)));
                    break;
                case HITS:
                    trainingHITSVertexIndexesMap.put(vertexIndex, new Vertex<>(new HITSAlgorithm.VertexContentType(vertexIndex, 1.0, 1.0)));
                    break;
            }
            Vertex<GraphRecursiveNeuralNetwork.VertexContentType, Void> rnnGraphVertex =
                    new Vertex<>(new GraphRecursiveNeuralNetwork.VertexContentType(vertexIndex, 0, null, null));
            trainingRNNVertexIndexesMap.put(vertexIndex, rnnGraphVertex);
        }
        for (int vertexIndex : testingDataSet.getVertexIndices()) {
            switch (rankingAlgorithm) {
                case PAGERANK:
                    testingPRVertexIndexesMap.put(vertexIndex, new Vertex<>(new PageRankAlgorithm.VertexContentType(vertexIndex, 0.0)));
                    break;
                case HITS:
                    testingHITSVertexIndexesMap.put(vertexIndex, new Vertex<>(new HITSAlgorithm.VertexContentType(vertexIndex, 1.0, 1.0)));
                    break;
            }
            Vertex<GraphRecursiveNeuralNetwork.VertexContentType, Void> rnnGraphVertex =
                    new Vertex<>(new GraphRecursiveNeuralNetwork.VertexContentType(vertexIndex, 0, null, null));
            testingRNNVertexIndexesMap.put(vertexIndex, rnnGraphVertex);
        }
        for (DataSets.Edge edge : trainingDataSet.getEdges()) {
            switch (rankingAlgorithm) {
                case PAGERANK:
                    trainingPRGraph.addEdge(trainingPRVertexIndexesMap.get(edge.getSourceVertexIndex()), trainingPRVertexIndexesMap.get(edge.getDestinationVertexIndex()));
                    break;
                case HITS:
                    trainingHITSGraph.addEdge(trainingHITSVertexIndexesMap.get(edge.getSourceVertexIndex()), trainingHITSVertexIndexesMap.get(edge.getDestinationVertexIndex()));
                    break;
            }
            trainingRNNGraph.addEdge(trainingRNNVertexIndexesMap.get(edge.getSourceVertexIndex()), trainingRNNVertexIndexesMap.get(edge.getDestinationVertexIndex()));
        }
        for (DataSets.Edge edge : testingDataSet.getEdges()) {
            switch (rankingAlgorithm) {
                case PAGERANK:
                    testingPRGraph.addEdge(testingPRVertexIndexesMap.get(edge.getSourceVertexIndex()), testingPRVertexIndexesMap.get(edge.getDestinationVertexIndex()));
                    break;
                case HITS:
                    testingHITSGraph.addEdge(testingHITSVertexIndexesMap.get(edge.getSourceVertexIndex()), testingHITSVertexIndexesMap.get(edge.getDestinationVertexIndex()));
                    break;
            }
            testingRNNGraph.addEdge(testingRNNVertexIndexesMap.get(edge.getSourceVertexIndex()), testingRNNVertexIndexesMap.get(edge.getDestinationVertexIndex()));
        }
        logger.info("Finished generating graphs for the " + trainingDataSetName + " data set and for the " + testingDataSetName + " data set.");
        this.featureVectorFunctionType = featureVectorFunctionType;
        this.evaluationResultsFolderPath = evaluationResultsFolderPath;
        switch (rankingAlgorithm) {
            case PAGERANK:
                logger.info("Running PageRank for the " + trainingDataSetName + " data set.");
                PageRankAlgorithm trainingPR =
                        new PageRankAlgorithm.Builder<>(trainingPRGraph)
                                .dampingFactor(0.85)
                                .maximumNumberOfIterations(1000)
                                .checkForRankConvergence(false)
                                .loggingLevel(2)
                                .build();
                trainingPR.computeRanks();
                logger.info("Finished running PageRank for the " + trainingDataSetName + " data set.");
                for (Vertex<PageRankAlgorithm.VertexContentType, Void> vertex : trainingPRGraph.getVertices())
                    trueTrainingScores.put(vertex.getContent().getId(), Vectors.dense((double) vertex.getContent().getRank()));
                logger.info("Running PageRank for the " + testingDataSetName + " data set.");
                PageRankAlgorithm testingPR =
                        new PageRankAlgorithm.Builder<>(testingPRGraph)
                                .dampingFactor(0.85)
                                .maximumNumberOfIterations(1000)
                                .checkForRankConvergence(false)
                                .loggingLevel(2)
                                .build();
                testingPR.computeRanks();
                logger.info("Finished running PageRank for the " + testingDataSetName + " data set.");
                for (Vertex<PageRankAlgorithm.VertexContentType, Void> vertex : testingPRGraph.getVertices())
                    trueTestingScores.put(vertex.getContent().getId(), Vectors.dense((double) vertex.getContent().getRank()));
                break;
            case HITS:
                logger.info("Running HITS for the " + trainingDataSetName + " data set.");
                HITSAlgorithm trainingHITS =
                        new HITSAlgorithm.Builder<>(trainingHITSGraph)
                                .maximumNumberOfIterations(1000)
                                .checkForScoresConvergence(false)
                                .loggingLevel(2)
                                .build();
                trainingHITS.computeScores();
                logger.info("Finished running HITS for the " + trainingDataSetName + " data set.");
                for (Vertex<HITSAlgorithm.VertexContentType, Void> vertex : trainingHITSGraph.getVertices())
                    trueTrainingScores.put(vertex.getContent().getId(), Vectors.dense(vertex.getContent().getAuthorityScore(),
                                                                                      vertex.getContent().getHubScore()));
                logger.info("Running HITS for the " + testingDataSetName + " data set.");
                HITSAlgorithm testingHITS =
                        new HITSAlgorithm.Builder<>(testingHITSGraph)
                                .maximumNumberOfIterations(1000)
                                .checkForScoresConvergence(false)
                                .loggingLevel(2)
                                .build();
                testingHITS.computeScores();
                logger.info("Finished running HITS for the " + testingDataSetName + " data set.");
                for (Vertex<HITSAlgorithm.VertexContentType, Void> vertex : testingHITSGraph.getVertices())
                    trueTestingScores.put(vertex.getContent().getId(), Vectors.dense(vertex.getContent().getAuthorityScore(),
                                                                                     vertex.getContent().getHubScore()));
                break;
        }
    }

    public void runRNNExperiments(int[] featureVectorsSizes, int[] maximumNumberOfStepsToTry) {
        EvaluationResults[][] results = new EvaluationResults[featureVectorsSizes.length][maximumNumberOfStepsToTry.length];
        for (int i = 0; i < featureVectorsSizes.length; i++)
            for (int j = 0; j < maximumNumberOfStepsToTry.length; j++) {
                results[i][j] = runRNNExperiment(featureVectorsSizes[i], maximumNumberOfStepsToTry[j]);
                appendEvaluationResultsToFile(featureVectorsSizes[i],
                                              maximumNumberOfStepsToTry[j],
                                              results[i][j],
                                              evaluationResultsFolderPath);
            }
        logger.info("Results for the " + testingDataSetName + " data set:");
        for (int i = 0; i < featureVectorsSizes.length; i++)
            for (int j = 0; j < maximumNumberOfStepsToTry.length; j++)
                logger.info("\tF = %d, K = %d:\t { %s }",
                            featureVectorsSizes[i],
                            maximumNumberOfStepsToTry[j],
                            results[i][j].toString());
    }

    private EvaluationResults runRNNExperiment(int featureVectorsSize, int maximumNumberOfSteps) {
        int outputSize = 1;
        if (rankingAlgorithm.equals(RankingAlgorithm.HITS))
            outputSize = 2;
        VertexRankingRecursiveNeuralNetwork<Void> trainGRNN =
                new VertexRankingRecursiveNeuralNetwork<>(featureVectorsSize, outputSize, maximumNumberOfSteps, trainingRNNGraph, trueTrainingScores, featureVectorFunctionType);
//        if (!graphRNNAlgorithm.checkDerivative(1e-5))
//            logger.warn("The derivatives of the RNN objective function provided are not the same as those obtained " +
//                                "by the method of finite differences.");
        logger.info("Training RNN for the " + trainingDataSetName + " data set.");
        trainGRNN.trainNetwork();
        trainGRNN.performForwardPass();
        logger.info("Finished training RNN for the " + trainingDataSetName + " data set.");
        logger.info("Testing RNN for the " + testingDataSetName + " data set.");
        VertexRankingRecursiveNeuralNetwork<Void> testGRNN =
                new VertexRankingRecursiveNeuralNetwork<>(featureVectorsSize, outputSize, maximumNumberOfSteps, testingRNNGraph, trueTestingScores, featureVectorFunctionType);
        testGRNN.setOutputFunctionParameters(trainGRNN.getOutputFunctionParameters());
        testGRNN.setFeatureVectorFunctionParameters(trainGRNN.getFeatureVectorFunctionParameters());
        testGRNN.performForwardPass();
        logger.info("Finished testing RNN for the " + testingDataSetName + " data set.");
        logger.info("Storing RNN results for the " + testingDataSetName + " data set.");
        Map<Integer, Vector> predictions = new HashMap<>();
        testingRNNGraph.getVertices()
                .stream()
                .filter(vertex -> trueTestingScores.containsKey(vertex.getContent().getId()))
                .forEach(vertex -> predictions.put(vertex.getContent().getId(),
                                                   testGRNN.getOutputForVertex(vertex)));
        logger.info("Finished storing RNN results for the " + trainingDataSetName + " data set.");
        EvaluationResults results = evaluateResults(predictions);
        trainGRNN.resetGraph();
        return results;
    }

    private EvaluationResults evaluateResults(Map<Integer, Vector> predictions) {
        final int[] ranks = new int[] { 10, 100, 1000, trueTestingScores.size() };
        final double[] overallRMSE = new double[ranks.length];
        final double[] overallMin = new double[ranks.length];
        final double[] overallMax = new double[ranks.length];
        for (int rank = 0; rank < ranks.length; rank++) {
            overallMin[rank] = Double.MAX_VALUE;
            overallMax[rank] = -Double.MAX_VALUE;
        }
        int currentOverallRank = 0;
        int currentOverallScore = 1;
        Map<Integer, Double> sortingMap = new HashMap<>();
        for (Map.Entry<Integer, Vector> trueScore : trueTestingScores.entrySet())
            sortingMap.put(trueScore.getKey(), trueScore.getValue().sum());
        sortingMap = CollectionUtilities.sortByValue(sortingMap);
        for (int vertexId : sortingMap.keySet()) {
            double loss = Math.abs(predictions.get(vertexId).sum() - trueTestingScores.get(vertexId).sum());
            if (currentOverallScore++ > ranks[currentOverallRank])
                currentOverallRank++;
            for (int rank = ranks.length - 1; rank >= currentOverallRank; rank--) {
                overallRMSE[rank] += loss;
                overallMin[rank] = Math.min(overallMin[rank], trueTestingScores.get(vertexId).sum());
                overallMax[rank] = Math.max(overallMax[rank], trueTestingScores.get(vertexId).sum());
            }
        }
        for (int rank = 0; rank < ranks.length; rank++)
            if (rank < ranks.length - 1)
                overallRMSE[rank] /= ranks[rank];
            else
                overallRMSE[rank] /= currentOverallScore - 1;
        return new EvaluationResults(ranks, overallRMSE, overallMin, overallMax);
    }

    private static class EvaluationResults {
        private final int[] ranks;
        private final double[] overallRMSE;
        private final double[] overallMin;
        private final double[] overallMax;

        private EvaluationResults(int[] ranks, double[] overallRMSE, double[] overallMin, double[] overallMax) {
            this.ranks = ranks;
            this.overallRMSE = overallRMSE;
            this.overallMin = overallMin;
            this.overallMax = overallMax;
        }

        public String toString() {
            return "Ranks:" + arrayToString(ranks) + "\t" +
                    arrayToString(overallRMSE) + "|" +
                    arrayToString(overallMin) + "|" +
                    arrayToString(overallMax);
        }

        private static String arrayToString(int[] array) {
            StringJoiner stringJoiner = new StringJoiner(",");
            for (int element : array)
                stringJoiner.add(String.valueOf(element));
            return stringJoiner.toString();
        }

        private static String arrayToString(double[] array) {
            StringJoiner stringJoiner = new StringJoiner(",");
            for (double element : array)
                stringJoiner.add(String.valueOf(element));
            return stringJoiner.toString();
        }
    }

    private void appendEvaluationResultsToFile(int featureVectorsSize,
                                               int maximumNumberOfSteps,
                                               EvaluationResults results,
                                               String folderPath) {
        try {
            String resultsLine = featureVectorsSize + "\t" + maximumNumberOfSteps + "\t" + results.toString() + "\n";
            Files.write(Paths.get(folderPath + "/results_" + normalizeDataSetName(trainingDataSetName)  + "_" + normalizeDataSetName(testingDataSetName) + "_" + featureVectorFunctionType.name().toLowerCase() + ".txt"),
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

    private enum RankingAlgorithm {
        PAGERANK,
        HITS;
    }

    public static void main(String[] args) {
        RankingAlgorithm rankingAlgorithm = RankingAlgorithm.valueOf(args[0]);
        DataSets.DataSet trainingDataSet = DataSets.loadUnlabeledDataSet(args[4], args[6].equals("1"));
        DataSets.DataSet testingDataSet = DataSets.loadUnlabeledDataSet(args[5], args[6].equals("1"));
        VertexRankingExperiment2 experiment = new VertexRankingExperiment2(trainingDataSet,
                                                                           testingDataSet,
                                                                           rankingAlgorithm,
                                                                           FeatureVectorFunctionType.valueOf(args[1]),
                                                                           args[7]);
        experiment.runRNNExperiments(intArrayFromString(args[2]), intArrayFromString(args[3]));
    }

    private static int[] intArrayFromString(String string) {
        String[] stringParts = string.split(",");
        int[] array = new int[stringParts.length];
        for (int partIndex = 0; partIndex < stringParts.length; partIndex++)
            array[partIndex] = Integer.parseInt(stringParts[partIndex]);
        return array;
    }
}
