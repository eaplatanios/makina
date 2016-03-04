package org.platanios.experiment.graph;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.graph.Graph;
import org.platanios.learn.graph.HITSAlgorithm;
import org.platanios.learn.graph.Vertex;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.neural.graph.FeatureVectorFunctionType;
import org.platanios.learn.neural.graph.GraphRecursiveNeuralNetwork;
import org.platanios.learn.neural.graph.VertexRankingRecursiveNeuralNetwork;

import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DeepHITSExperiment {
    private final Logger logger = LogManager.getFormatterLogger("Deep HITS Experiment");
    private final Map<Integer, Vertex<HITSAlgorithm.VertexContentType, Void>> hitsVertexIndexesMap = new HashMap<>();
    private final Map<Integer, Vertex<GraphRecursiveNeuralNetwork.VertexContentType, Void>> rnnVertexIndexesMap = new HashMap<>();
    private final Map<Integer, Vector> trainingData = new HashMap<>();
    private final Map<Integer, Score> hitsScores = new TreeMap<>();
    private final Map<Integer, Score> rnnScores = new TreeMap<>();

    private final String dataSetName;
    private final Graph<HITSAlgorithm.VertexContentType, Void> hitsGraph;
    private final Graph<GraphRecursiveNeuralNetwork.VertexContentType, Void> rnnGraph;

    private VertexRankingRecursiveNeuralNetwork<Void> graphRNNAlgorithm;

    public DeepHITSExperiment(DataSets.DataSet dataSet) {
        dataSetName = dataSet.getName();
        logger.info("Number of vertices: " + dataSet.getVertexIndices().size());
        logger.info("Number of edges: " + dataSet.getEdges().size());
        hitsGraph = new Graph<>();
        rnnGraph = new Graph<>();
        for (int vertexIndex : dataSet.getVertexIndices()) {
            hitsVertexIndexesMap.put(vertexIndex, new Vertex<>(new HITSAlgorithm.VertexContentType(vertexIndex, 1.0, 1.0)));
            Vertex<GraphRecursiveNeuralNetwork.VertexContentType, Void> rnnGraphVertex =
                    new Vertex<>(new GraphRecursiveNeuralNetwork.VertexContentType(vertexIndex, 0, null, null, null, null));
            rnnVertexIndexesMap.put(vertexIndex, rnnGraphVertex);
        }
        for (DataSets.Edge edge : dataSet.getEdges()) {
            hitsGraph.addEdge(hitsVertexIndexesMap.get(edge.getSourceVertexIndex()), hitsVertexIndexesMap.get(edge.getDestinationVertexIndex()));
            rnnGraph.addEdge(rnnVertexIndexesMap.get(edge.getSourceVertexIndex()), rnnVertexIndexesMap.get(edge.getDestinationVertexIndex()));
        }
        logger.info("Finished generating graphs for the " + dataSetName + " data set.");
    }

    public void runHITS() {
        logger.info("Running HITS for the " + dataSetName + " data set.");
        HITSAlgorithm hitsAlgorithm =
                new HITSAlgorithm.Builder<>(hitsGraph)
                        .maximumNumberOfIterations(1000)
                        .checkForScoresConvergence(false)
                        .loggingLevel(5)
                        .build();
        hitsAlgorithm.computeScores();
        for (Vertex<HITSAlgorithm.VertexContentType, Void> vertex : hitsGraph.getVertices())
            hitsScores.put(vertex.getContent().getId(), new Score(vertex.getContent().getAuthorityScore(), vertex.getContent().getHubScore()));
        logger.info("Finished running HITS for the " + dataSetName + " data set.");
    }

    public void sampleRNNTrainingData(double percentageOfTrainingData) {
        logger.info("Sampling RNN training data for the " + dataSetName + " data set.");
        final List<Map.Entry<Integer, Score>> entries = new ArrayList<>(hitsScores.entrySet());
        Collections.shuffle(entries);
        for (int sample = 0; sample < Math.floor(percentageOfTrainingData * hitsScores.size()); sample++)
            trainingData.put(entries.get(sample).getKey(), Vectors.dense(entries.get(sample).getValue().authorityScore, entries.get(sample).getValue().hubScore));
        logger.info("Finished sampling RNN training data for the " + dataSetName + " data set.");
    }

    public void runRNNExperiments(int[] featureVectorsSizes, int[] maximumNumberOfStepsToTry) {
        EvaluationResults[][] results = new EvaluationResults[featureVectorsSizes.length][maximumNumberOfStepsToTry.length];
        for (int i = 0; i < featureVectorsSizes.length; i++)
            for (int j = 0; j < maximumNumberOfStepsToTry.length; j++)
                results[i][j] = runRNNExperiment(featureVectorsSizes[i], maximumNumberOfStepsToTry[j]);
        logger.info("Results for the " + dataSetName + " data set:");
        for (int i = 0; i < featureVectorsSizes.length; i++)
            for (int j = 0; j < maximumNumberOfStepsToTry.length; j++)
                logger.info("\tF = %d, K = %d:\t { Train MSE: %20s | Test MSE: %20s | Overall MSE: %20s }",
                            featureVectorsSizes[i],
                            maximumNumberOfStepsToTry[j],
                            results[i][j].trainMSE,
                            results[i][j].testMSE,
                            results[i][j].overallMSE);
    }

    public EvaluationResults runRNNExperiment(int featureVectorsSize, int maximumNumberOfSteps) {
        graphRNNAlgorithm = new VertexRankingRecursiveNeuralNetwork<>(featureVectorsSize, 2, maximumNumberOfSteps, rnnGraph, trainingData, FeatureVectorFunctionType.SIGMOID);
        if (!graphRNNAlgorithm.checkDerivative(1e-5))
            logger.warn("The derivatives of the RNN objective function provided are not the same as those obtained " +
                                "by the method of finite differences.");
        logger.info("Training RNN for the " + dataSetName + " data set.");
        graphRNNAlgorithm.trainNetwork();
        graphRNNAlgorithm.performForwardPass();
        logger.info("Finished training RNN for the " + dataSetName + " data set.");
        logger.info("Storing RNN results for the " + dataSetName + " data set.");
        for (Vertex<GraphRecursiveNeuralNetwork.VertexContentType, Void> vertex : rnnGraph.getVertices())
            rnnScores.put(vertex.getContent().getId(), new Score(graphRNNAlgorithm.getOutputForVertex(vertex).get(0),
                                                                 graphRNNAlgorithm.getOutputForVertex(vertex).get(1)));
        logger.info("Finished storing RNN results for the " + dataSetName + " data set.");
        EvaluationResults results = evaluateResults();
        graphRNNAlgorithm.resetGraph();
        return results;
    }

    private EvaluationResults evaluateResults() {
        double trainMSE = 0.0;
        double testMSE = 0.0;
        double overallMSE = 0.0;
        for (int vertexID : hitsScores.keySet()) {
            double loss = Math.pow(rnnScores.get(vertexID).authorityScore - hitsScores.get(vertexID).authorityScore, 2)
                    + Math.pow(rnnScores.get(vertexID).hubScore - hitsScores.get(vertexID).hubScore, 2);
            if (trainingData.containsKey(vertexID))
                trainMSE += loss;
            else
                testMSE += loss;
            overallMSE += loss;
        }
        trainMSE /= trainingData.size();
        testMSE /= (hitsScores.size() - trainingData.size());
        overallMSE /= hitsScores.size();
        return new EvaluationResults(trainMSE, testMSE, overallMSE);
    }

    public static class EvaluationResults {
        private final double trainMSE;
        private final double testMSE;
        private final double overallMSE;

        public EvaluationResults(double trainMSE, double testMSE, double overallMSE) {
            this.trainMSE = trainMSE;
            this.testMSE = testMSE;
            this.overallMSE = overallMSE;
        }

        public double getTrainMSE() {
            return trainMSE;
        }

        public double getTestMSE() {
            return testMSE;
        }

        public double getOverallMSE() {
            return overallMSE;
        }
    }

    private static class Score {
        private final double authorityScore;
        private final double hubScore;

        public Score(double authorityScore, double hubScore) {
            this.authorityScore = authorityScore;
            this.hubScore = hubScore;
        }
    }

    public static void main(String[] args) {
//        DataSets.DataSet dataSet = DataSets.loadUnlabeledDataSet("/Users/Anthony/Downloads/Pokec Graph Data Set/soc-pokec-relationships.txt");
//        DataSets.DataSet dataSet = DataSets.loadTwitterSocialCirclesDataSet("/Users/Anthony/Downloads/Twitter Social Circles Data Set/twitter_combined.txt");
//        DataSets.DataSet dataSet = DataSets.loadUnlabeledDataSet("/Users/Anthony/Downloads/soc-Epinions1.txt");
//        DataSets.DataSet dataSet = DataSets.loadTwitterSocialCirclesDataSet("/Users/Anthony/Downloads/facebook_combined.txt");
//        DataSets.DataSet dataSet;
//        if (args[1].equals("0"))
//            dataSet = DataSets.loadUnlabeledDataSet(args[0]);
//        else
//            dataSet = DataSets.loadTwitterSocialCirclesDataSet(args[0]);
//        DeepHITSExperiment experiment = new DeepHITSExperiment(dataSet);
//        experiment.runHITS();
//        experiment.sampleRNNTrainingData(0.8);
//        experiment.runRNNExperiments(new int[] { 1, 10 }, new int[] { 2, 11, 21, 31, 41, 51 });
    }
}
