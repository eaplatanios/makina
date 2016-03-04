package org.platanios.experiment.graph;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.graph.Graph;
import org.platanios.learn.graph.HITSAlgorithm;
import org.platanios.learn.graph.PageRankAlgorithm;
import org.platanios.learn.graph.Vertex;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
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
public class VertexRankingExperiment {
    private final Logger logger = LogManager.getFormatterLogger("Vertex Ranking Experiment");
    private final Map<Integer, Vertex<PageRankAlgorithm.VertexContentType, Void>> prVertexIndexesMap = new HashMap<>();
    private final Map<Integer, Vertex<HITSAlgorithm.VertexContentType, Void>> hitsVertexIndexesMap = new HashMap<>();
    private final Map<Integer, Vertex<GraphRecursiveNeuralNetwork.VertexContentType, Void>> rnnVertexIndexesMap = new HashMap<>();
    private final Map<Integer, Vector> trueScores = new TreeMap<>();

    private final String dataSetName;
    private final RankingAlgorithm rankingAlgorithm;
    private final Graph<PageRankAlgorithm.VertexContentType, Void> prGraph;
    private final Graph<HITSAlgorithm.VertexContentType, Void> hitsGraph;
    private final Graph<GraphRecursiveNeuralNetwork.VertexContentType, Void> rnnGraph;
    private final FeatureVectorFunctionType featureVectorFunctionType;
    private final String evaluationResultsFolderPath;

    public VertexRankingExperiment(DataSets.DataSet dataSet,
                                   RankingAlgorithm rankingAlgorithm,
                                   FeatureVectorFunctionType featureVectorFunctionType,
                                   String evaluationResultsFolderPath) {
        dataSetName = dataSet.getName();
        this.rankingAlgorithm = rankingAlgorithm;
        logger.info("Number of vertices: " + dataSet.getVertexIndices().size());
        logger.info("Number of edges: " + dataSet.getEdges().size());
        switch (rankingAlgorithm) {
            case PAGERANK:
                prGraph = new Graph<>();
                hitsGraph = null;
                break;
            case HITS:
                prGraph = null;
                hitsGraph = new Graph<>();
                break;
            default:
                prGraph = null;
                hitsGraph = null;
                break;
        }
        rnnGraph = new Graph<>();
        for (int vertexIndex : dataSet.getVertexIndices()) {
            switch (rankingAlgorithm) {
                case PAGERANK:
                    prVertexIndexesMap.put(vertexIndex, new Vertex<>(new PageRankAlgorithm.VertexContentType(vertexIndex, 0.0)));
                    break;
                case HITS:
                    hitsVertexIndexesMap.put(vertexIndex, new Vertex<>(new HITSAlgorithm.VertexContentType(vertexIndex, 1.0, 1.0)));
                    break;
            }
            Vertex<GraphRecursiveNeuralNetwork.VertexContentType, Void> rnnGraphVertex =
                    new Vertex<>(new GraphRecursiveNeuralNetwork.VertexContentType(vertexIndex, 0, null, null));
            rnnVertexIndexesMap.put(vertexIndex, rnnGraphVertex);
        }
        for (DataSets.Edge edge : dataSet.getEdges()) {
            switch (rankingAlgorithm) {
                case PAGERANK:
                    prGraph.addEdge(prVertexIndexesMap.get(edge.getSourceVertexIndex()), prVertexIndexesMap.get(edge.getDestinationVertexIndex()));
                    break;
                case HITS:
                    hitsGraph.addEdge(hitsVertexIndexesMap.get(edge.getSourceVertexIndex()), hitsVertexIndexesMap.get(edge.getDestinationVertexIndex()));
                    break;
            }
            rnnGraph.addEdge(rnnVertexIndexesMap.get(edge.getSourceVertexIndex()), rnnVertexIndexesMap.get(edge.getDestinationVertexIndex()));
        }
        logger.info("Finished generating graphs for the " + dataSetName + " data set.");
        this.featureVectorFunctionType = featureVectorFunctionType;
        this.evaluationResultsFolderPath = evaluationResultsFolderPath;
        logger.info("Running PageRank for the " + dataSetName + " data set.");
        switch (rankingAlgorithm) {
            case PAGERANK:
                PageRankAlgorithm pageRankAlgorithm =
                        new PageRankAlgorithm.Builder<>(prGraph)
                                .dampingFactor(0.85)
                                .maximumNumberOfIterations(1000)
                                .checkForRankConvergence(false)
                                .loggingLevel(5)
                                .build();
                pageRankAlgorithm.computeRanks();
                for (Vertex<PageRankAlgorithm.VertexContentType, Void> vertex : prGraph.getVertices())
                    trueScores.put(vertex.getContent().getId(), Vectors.dense((double) vertex.getContent().getRank()));
                break;
            case HITS:
                HITSAlgorithm hitsAlgorithm =
                        new HITSAlgorithm.Builder<>(hitsGraph)
                                .maximumNumberOfIterations(1000)
                                .checkForScoresConvergence(false)
                                .loggingLevel(5)
                                .build();
                hitsAlgorithm.computeScores();
                for (Vertex<HITSAlgorithm.VertexContentType, Void> vertex : hitsGraph.getVertices())
                    trueScores.put(vertex.getContent().getId(), Vectors.dense(vertex.getContent().getAuthorityScore(),
                                                                              vertex.getContent().getHubScore()));
                break;
        }
        logger.info("Finished running PageRank for the " + dataSetName + " data set.");
    }

    public void runRNNExperiments(int[] featureVectorsSizes,
                                  int[] maximumNumberOfStepsToTry,
                                  double trainingExamplesProportion) {
        EvaluationResults[][] results = new EvaluationResults[featureVectorsSizes.length][maximumNumberOfStepsToTry.length];
        for (int i = 0; i < featureVectorsSizes.length; i++)
            for (int j = 0; j < maximumNumberOfStepsToTry.length; j++) {
                final List<Map.Entry<Integer, Vector>> entries = new ArrayList<>(trueScores.entrySet());
                Collections.shuffle(entries);
                Map<Integer, Vector> trainingData = new HashMap<>();
                for (Map.Entry<Integer, Vector> entry : entries.subList(0, (int) Math.floor(trainingExamplesProportion * entries.size())))
                    trainingData.put(entry.getKey(), entry.getValue());
                results[i][j] = runRNNExperiment(
                        featureVectorsSizes[i],
                        maximumNumberOfStepsToTry[j],
                        trainingData
                );
                appendEvaluationResultsToFile(featureVectorsSizes[i],
                                              maximumNumberOfStepsToTry[j],
                                              results[i][j],
                                              evaluationResultsFolderPath);
            }
        logger.info("Results for the " + dataSetName + " data set:");
        for (int i = 0; i < featureVectorsSizes.length; i++)
            for (int j = 0; j < maximumNumberOfStepsToTry.length; j++)
                logger.info("\tF = %d, K = %d:\t { %s }",
                            featureVectorsSizes[i],
                            maximumNumberOfStepsToTry[j],
                            results[i][j].toString());
    }

    private EvaluationResults runRNNExperiment(int featureVectorsSize,
                                               int maximumNumberOfSteps,
                                               Map<Integer, Vector> trainingData) {
        int outputSize = 1;
        if (rankingAlgorithm.equals(RankingAlgorithm.HITS))
            outputSize = 2;
        VertexRankingRecursiveNeuralNetwork<Void> graphRNNAlgorithm =
                new VertexRankingRecursiveNeuralNetwork<>(featureVectorsSize, outputSize, maximumNumberOfSteps, rnnGraph, trainingData, featureVectorFunctionType);
        if (!graphRNNAlgorithm.checkDerivative(1e-5))
            logger.warn("The derivatives of the RNN objective function provided are not the same as those obtained " +
                                "by the method of finite differences.");
        logger.info("Training RNN for the " + dataSetName + " data set.");
        graphRNNAlgorithm.trainNetwork();
        graphRNNAlgorithm.performForwardPass();
        logger.info("Finished training RNN for the " + dataSetName + " data set.");
        logger.info("Storing RNN results for the " + dataSetName + " data set.");
        Map<Integer, Vector> predictions = new HashMap<>();
        rnnGraph.getVertices()
                .stream()
                .filter(vertex -> trueScores.containsKey(vertex.getContent().getId()))
                .forEach(vertex -> predictions.put(vertex.getContent().getId(),
                                                   graphRNNAlgorithm.getOutputForVertex(vertex)));
        logger.info("Finished storing RNN results for the " + dataSetName + " data set.");
        EvaluationResults results = evaluateResults(predictions, trainingData);
        graphRNNAlgorithm.resetGraph();
        return results;
    }

    private EvaluationResults evaluateResults(Map<Integer, Vector> predictions, Map<Integer, Vector> trainingData) {
        final int[] ranks = new int[] { 10, 100, 1000, trueScores.size() };
        final double[] trainRMSE = new double[ranks.length];
        final double[] trainMin = new double[ranks.length];
        final double[] trainMax = new double[ranks.length];
        final double[] testRMSE = new double[ranks.length];
        final double[] testMin = new double[ranks.length];
        final double[] testMax = new double[ranks.length];
        final double[] overallRMSE = new double[ranks.length];
        final double[] overallMin = new double[ranks.length];
        final double[] overallMax = new double[ranks.length];
        for (int rank = 0; rank < ranks.length; rank++) {
            trainMin[rank] = Double.MAX_VALUE;
            trainMax[rank] = -Double.MAX_VALUE;
            testMin[rank] = Double.MAX_VALUE;
            testMax[rank] = -Double.MAX_VALUE;
            overallMin[rank] = Double.MAX_VALUE;
            overallMax[rank] = -Double.MAX_VALUE;
        }
        int currentTrainRank = 0;
        int currentTestRank = 0;
        int currentOverallRank = 0;
        int currentTrainScore = 1;
        int currentTestScore = 1;
        int currentOverallScore = 1;
        Map<Integer, Double> sortingMap = new HashMap<>();
        for (Map.Entry<Integer, Vector> trueScore : trueScores.entrySet())
            sortingMap.put(trueScore.getKey(), trueScore.getValue().sum());
        sortingMap = CollectionUtilities.sortByValue(sortingMap);
        for (int vertexId : sortingMap.keySet()) {
            double loss = Math.abs(predictions.get(vertexId).sum() - trueScores.get(vertexId).sum());
            if (trainingData.containsKey(vertexId)) {
                if (currentTrainScore++ > ranks[currentTrainRank])
                    currentTrainRank++;
                for (int rank = ranks.length - 1; rank >= currentTrainRank; rank--) {
                    trainRMSE[rank] += loss;
                    trainMin[rank] = Math.min(trainMin[rank], trueScores.get(vertexId).sum());
                    trainMax[rank] = Math.max(trainMax[rank], trueScores.get(vertexId).sum());
                }
            } else {
                if (currentTestScore++ > ranks[currentTestRank])
                    currentTestRank++;
                for (int rank = ranks.length - 1; rank >= currentTestRank; rank--) {
                    testRMSE[rank] += loss;
                    testMin[rank] = Math.min(testMin[rank], trueScores.get(vertexId).sum());
                    testMax[rank] = Math.max(testMax[rank], trueScores.get(vertexId).sum());
                }
            }
            if (currentOverallScore++ > ranks[currentOverallRank])
                currentOverallRank++;
            for (int rank = ranks.length - 1; rank >= currentOverallRank; rank--) {
                overallRMSE[rank] += loss;
                overallMin[rank] = Math.min(overallMin[rank], trueScores.get(vertexId).sum());
                overallMax[rank] = Math.max(overallMax[rank], trueScores.get(vertexId).sum());
            }
        }
        for (int rank = 0; rank < ranks.length; rank++) {
            if (rank < ranks.length - 1) {
                trainRMSE[rank] /= ranks[rank];
                testRMSE[rank] /= ranks[rank];
                overallRMSE[rank] /= ranks[rank];
            } else {
                trainRMSE[rank] /= currentTrainScore - 1;
                testRMSE[rank] /= currentTestScore - 1;
                overallRMSE[rank] /= currentOverallScore - 1;
            }
//            trainRMSE[rank] = Math.sqrt(trainRMSE[rank]);
//            testRMSE[rank] = Math.sqrt(testRMSE[rank]);
//            overallRMSE[rank] = Math.sqrt(overallRMSE[rank]);
        }
        return new EvaluationResults(ranks,
                                     trainRMSE,
                                     trainMin,
                                     trainMax,
                                     testRMSE,
                                     testMin,
                                     testMax,
                                     overallRMSE,
                                     overallMin,
                                     overallMax);
    }

    private static class EvaluationResults {
        private final int[] ranks;
        private final double[] trainRMSE;
        private final double[] trainMin;
        private final double[] trainMax;
        private final double[] testRMSE;
        private final double[] testMin;
        private final double[] testMax;
        private final double[] overallRMSE;
        private final double[] overallMin;
        private final double[] overallMax;

        private EvaluationResults(int[] ranks,
                                  double[] trainRMSE,
                                  double[] trainMin,
                                  double[] trainMax,
                                  double[] testRMSE,
                                  double[] testMin,
                                  double[] testMax,
                                  double[] overallRMSE,
                                  double[] overallMin,
                                  double[] overallMax) {
            this.ranks = ranks;
            this.trainRMSE = trainRMSE;
            this.trainMin = trainMin;
            this.trainMax = trainMax;
            this.testRMSE = testRMSE;
            this.testMin = testMin;
            this.testMax = testMax;
            this.overallRMSE = overallRMSE;
            this.overallMin = overallMin;
            this.overallMax = overallMax;
        }

        public String toString() {
            return "Ranks:" + arrayToString(ranks) + "\t" +
                    "Train:" +
                    arrayToString(trainRMSE) + "|" +
                    arrayToString(trainMin) + "|" +
                    arrayToString(trainMax) + "\t" +
                    "Test:" +
                    arrayToString(testRMSE) + "|" +
                    arrayToString(testMin) + "|" +
                    arrayToString(testMax) + "\t" +
                    "Overall:" +
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

        private EvaluationResults averageResults(EvaluationResults... results) {
            final int[] ranks = results[0].ranks;
            final double[] trainRMSE = new double[ranks.length];
            final double[] trainMin = new double[ranks.length];
            final double[] trainMax = new double[ranks.length];
            final double[] testRMSE = new double[ranks.length];
            final double[] testMin = new double[ranks.length];
            final double[] testMax = new double[ranks.length];
            final double[] overallRMSE = new double[ranks.length];
            final double[] overallMin = new double[ranks.length];
            final double[] overallMax = new double[ranks.length];
            for (EvaluationResults result : results)
                for (int rank = 0; rank < ranks.length; rank++) {
                    trainRMSE[rank] += result.trainRMSE[rank];
                    trainMin[rank] += result.trainMin[rank];
                    trainMax[rank] += result.trainMax[rank];
                    testRMSE[rank] += result.testRMSE[rank];
                    testMin[rank] += result.testMin[rank];
                    testMax[rank] += result.testMax[rank];
                    overallRMSE[rank] += result.overallRMSE[rank];
                    overallMin[rank] += result.overallMin[rank];
                    overallMax[rank] += result.overallMax[rank];
                }
            for (int rank = 0; rank < ranks.length; rank++) {
                trainRMSE[rank] /= ranks.length;
                trainMin[rank] /= ranks.length;
                trainMax[rank] /= ranks.length;
                testRMSE[rank] /= ranks.length;
                testMin[rank] /= ranks.length;
                testMax[rank] /= ranks.length;
                overallRMSE[rank] /= ranks.length;
                overallMin[rank] /= ranks.length;
                overallMax[rank] /= ranks.length;
            }
            return new EvaluationResults(ranks,
                                         trainRMSE,
                                         trainMin,
                                         trainMax,
                                         testRMSE,
                                         testMin,
                                         testMax,
                                         overallRMSE,
                                         overallMin,
                                         overallMax);
        }
    }

    private void appendEvaluationResultsToFile(int featureVectorsSize,
                                               int maximumNumberOfSteps,
                                               EvaluationResults results,
                                               String folderPath) {
        try {
            String resultsLine = featureVectorsSize + "\t" + maximumNumberOfSteps + "\t" + results.toString() + "\n";
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

    private enum RankingAlgorithm {
        PAGERANK,
        HITS;
    }

    public static void main(String[] args) {
        RankingAlgorithm rankingAlgorithm = RankingAlgorithm.valueOf(args[0]);
        DataSets.DataSet dataSet = DataSets.loadUnlabeledDataSet(args[5], args[6].equals("1"));
        VertexRankingExperiment experiment = new VertexRankingExperiment(dataSet,
                                                                         rankingAlgorithm,
                                                                         FeatureVectorFunctionType.valueOf(args[1]),
                                                                         args[7]);
        experiment.runRNNExperiments(intArrayFromString(args[2]), intArrayFromString(args[3]), Double.parseDouble(args[4]));
    }

    private static int[] intArrayFromString(String string) {
        String[] stringParts = string.split(",");
        int[] array = new int[stringParts.length];
        for (int partIndex = 0; partIndex < stringParts.length; partIndex++)
            array[partIndex] = Integer.parseInt(stringParts[partIndex]);
        return array;
    }
}
