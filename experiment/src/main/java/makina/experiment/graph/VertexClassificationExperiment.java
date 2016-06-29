package makina.experiment.graph;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import makina.learn.evaluation.PrecisionRecall;
import makina.learn.graph.Graph;
import makina.learn.graph.Vertex;
import makina.learn.neural.graph.DeepGraph;
import makina.learn.neural.graph.UpdateFunctionType;
import makina.learn.neural.graph.VertexClassificationDeepGraph;
import makina.math.StatisticsUtilities;
import makina.math.matrix.Vector;
import makina.math.matrix.Vectors;
import makina.utilities.ArrayUtilities;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class VertexClassificationExperiment {
    private final Logger logger = LogManager.getFormatterLogger("Vertex Classification Experiment");
    private final Map<Integer, Vertex<DeepGraph.VertexContent, Void>> vertexIndexesMap = new HashMap<>();

    private final String dataSetName;
    private final Graph<DeepGraph.VertexContent, Void> graph;
    private final UpdateFunctionType updateFunctionType;
    private final Map<Integer, Integer> trueLabels;
    private final String evaluationResultsFolderPath;

    public VertexClassificationExperiment(DataSets.LabeledDataSet dataSet,
                                          UpdateFunctionType updateFunctionType,
                                          String evaluationResultsFolderPath) {
        dataSetName = dataSet.getName();
        logger.info("Number of vertices: " + dataSet.getVertexIndices().size());
        logger.info("Number of edges: " + dataSet.getEdges().size());
        graph = new Graph<>();
        for (int vertexIndex : dataSet.getVertexIndices())
            vertexIndexesMap.put(vertexIndex, new Vertex<>(new DeepGraph.VertexContent(vertexIndex, 0, null, null)));
        for (DataSets.Edge edge : dataSet.getEdges())
            graph.addEdge(vertexIndexesMap.get(edge.getSourceVertexIndex()), vertexIndexesMap.get(edge.getDestinationVertexIndex()));
        logger.info("Finished generating graphs for the " + dataSetName + " data set.");
        trueLabels = new HashMap<>(dataSet.getVertexLabels());
        this.updateFunctionType = updateFunctionType;
        this.evaluationResultsFolderPath = evaluationResultsFolderPath;
    }

    public void runRNNExperiments(int[] featureVectorsSizes,
                                  int[] maximumNumberOfStepsToTry,
                                  int numberOfFolds,
                                  int numberOfFoldsToKeep) {
        int numberOfClasses = new HashSet<>(trueLabels.values()).size();
        EvaluationResults[][] results = new EvaluationResults[featureVectorsSizes.length][maximumNumberOfStepsToTry.length];
        for (int i = 0; i < featureVectorsSizes.length; i++)
            for (int j = 0; j < maximumNumberOfStepsToTry.length; j++) {
                EvaluationResults[] currentSettingResults = new EvaluationResults[numberOfFoldsToKeep];
                final List<Map.Entry<Integer, Integer>> entries = new ArrayList<>(trueLabels.entrySet());
                Collections.shuffle(entries);
                int foldSize = Math.floorDiv(entries.size(), numberOfFolds);
                for (int foldNumber = 0; foldNumber < numberOfFolds; foldNumber++) {
                    Map<Integer, Vector> trainingData = new HashMap<>();
                    for (Map.Entry<Integer, Integer> entry : entries.subList(0, foldNumber * foldSize))
                        trainingData.put(entry.getKey(), Vectors.dense((double) entry.getValue()));
                    for (Map.Entry<Integer, Integer> entry : entries.subList((foldNumber + 1) * foldSize, entries.size()))
                        trainingData.put(entry.getKey(), Vectors.dense((double) entry.getValue()));
                    if (numberOfClasses > 2) {
                        currentSettingResults[foldNumber] = runMultiClassClassificationRNNExperiment(
                                featureVectorsSizes[i],
                                maximumNumberOfStepsToTry[j],
                                trainingData,
                                numberOfClasses
                        );
                    } else {
                        currentSettingResults[foldNumber] = runBinaryClassificationRNNExperiment(
                                featureVectorsSizes[i],
                                maximumNumberOfStepsToTry[j],
                                trainingData
                        );
                    }
                    if (foldNumber + 1 == numberOfFoldsToKeep)
                        break;
                }
                results[i][j] = currentSettingResults[0].averageResults(currentSettingResults);
                appendEvaluationResultsToFile(featureVectorsSizes[i],
                                              maximumNumberOfStepsToTry[j],
                                              results[i][j],
                                              evaluationResultsFolderPath);
            }
        logger.info("Results for the " + dataSetName + " data set:");
        for (int i = 0; i < featureVectorsSizes.length; i++)
            for (int j = 0; j < maximumNumberOfStepsToTry.length; j++)
                if (numberOfClasses > 2)
                    logger.info("\tF = %d, K = %d:\t { Train Acc.: %20s | Test Acc.: %20s | Overall Acc.: %20s }",
                                featureVectorsSizes[i],
                                maximumNumberOfStepsToTry[j],
                                ((BinaryClassificationEvaluationResults) results[i][j]).trainAUC,
                                ((BinaryClassificationEvaluationResults) results[i][j]).testAUC,
                                ((BinaryClassificationEvaluationResults) results[i][j]).overallAUC);
//                else
//                    logger.info("\tF = %d, K = %d:\t { Train Acc.: %20s | Test Acc.: %20s | Overall Acc.: %20s }",
//                                featureVectorsSizes[i],
//                                maximumNumberOfStepsToTry[j],
//                                ((MultiClassClassificationEvaluationResults) results[i][j]).trainAUC,
//                                ((MultiClassClassificationEvaluationResults) results[i][j]).testAUC,
//                                ((MultiClassClassificationEvaluationResults) results[i][j]).overallAUC);
    }

    public EvaluationResults runBinaryClassificationRNNExperiment(int featureVectorsSize,
                                                                  int maximumNumberOfSteps,
                                                                  Map<Integer, Vector> trainingData) {
        VertexClassificationDeepGraph<Void> graphRNNAlgorithm =
                new VertexClassificationDeepGraph<>(featureVectorsSize, 1, maximumNumberOfSteps, true, graph, updateFunctionType);
        logger.info("Training RNN for the " + dataSetName + " data set.");
        graphRNNAlgorithm.train(trainingData);
        logger.info("Finished training RNN for the " + dataSetName + " data set.");
        logger.info("Storing RNN results for the " + dataSetName + " data set.");
        List<TrueLabelPredictionPair> trainPairs = new ArrayList<>();
        List<TrueLabelPredictionPair> testPairs = new ArrayList<>();
        List<TrueLabelPredictionPair> allPairs = new ArrayList<>();
//        List<PredictedDataInstance<Vector, Integer>> trainPredictions = new ArrayList<>();
//        List<PredictedDataInstance<Vector, Integer>> testPredictions = new ArrayList<>();
//        List<PredictedDataInstance<Vector, Integer>> allPredictions = new ArrayList<>();
        graph.vertices()
                .stream()
                .filter(vertex -> trueLabels.containsKey(vertex.content().id()))
                .forEach(vertex -> {
//                    PredictedDataInstance<Vector, Integer> instance = new PredictedDataInstance<>(
//                            null,
//                            1,
//                            vertex.content().id(),
//                            graphRNNAlgorithm.output(vertex).get(0)
//                    );
                    if (trainingData.containsKey(vertex.content().id()))
                        trainPairs.add(new TrueLabelPredictionPair(trueLabels.get(vertex.content().id()) == 1,
                                                                   graphRNNAlgorithm.output(vertex).get(0)));
                    else
                        testPairs.add(new TrueLabelPredictionPair(trueLabels.get(vertex.content().id()) == 1,
                                                                  graphRNNAlgorithm.output(vertex).get(0)));
                    allPairs.add(new TrueLabelPredictionPair(trueLabels.get(vertex.content().id()) == 1,
                                                                 graphRNNAlgorithm.output(vertex).get(0)));
                });
        logger.info("Finished storing RNN results for the " + dataSetName + " data set.");
        EvaluationResults results = evaluateBinaryClassificationResults(trainPairs, testPairs, allPairs);
        graphRNNAlgorithm.resetGraph();
        return results;
    }

    public EvaluationResults runMultiClassClassificationRNNExperiment(int featureVectorsSize,
                                                                      int maximumNumberOfSteps,
                                                                      Map<Integer, Vector> trainingData,
                                                                      int numberOfClasses) {
        VertexClassificationDeepGraph<Void> graphRNNAlgorithm =
                new VertexClassificationDeepGraph<>(featureVectorsSize,
                                                    numberOfClasses,
                                                    maximumNumberOfSteps,
                                                    false,
                                                    graph,
                                                    updateFunctionType);
//        if (!graphRNNAlgorithm.checkDerivative(1e-5))
//            logger.warn("The derivatives of the RNN objective function provided are not the same as those obtained " +
//                                "by the method of finite differences.");
        logger.info("Training RNN for the " + dataSetName + " data set.");
        graphRNNAlgorithm.train(trainingData);
        logger.info("Finished training RNN for the " + dataSetName + " data set.");
        logger.info("Storing RNN results for the " + dataSetName + " data set.");
        Map<Integer, Integer> predictions = new HashMap<>();
        graph.vertices()
                .stream()
                .filter(vertex -> trueLabels.containsKey(vertex.content().id()))
                .forEach(vertex -> {
                    int maximumValueIndex = 0;
                    double maximumValue = 0.0;
                    for (Vector.Element element : graphRNNAlgorithm.output(vertex))
                        if (element.value() > maximumValue) {
                            maximumValueIndex = element.index();
                            maximumValue = element.value();
                        }
                    predictions.put(vertex.content().id(), maximumValueIndex);
                });
        logger.info("Finished storing RNN results for the " + dataSetName + " data set.");
        EvaluationResults results = evaluateMultiClassClassificationResults(predictions, trainingData, numberOfClasses);
        graphRNNAlgorithm.resetGraph();
        return results;
    }

    private class TrueLabelPredictionPair {
        private final boolean trueLabel;
        private final double prediction;

        public TrueLabelPredictionPair(boolean trueLabel, double prediction) {
            this.trueLabel = trueLabel;
            this.prediction = prediction;
        }

        public boolean trueLabel() {
            return trueLabel;
        }

        public double prediction() {
            return prediction;
        }
    }

    private EvaluationResults evaluateBinaryClassificationResults(
            List<TrueLabelPredictionPair> trainPairs,
            List<TrueLabelPredictionPair> testPairs,
            List<TrueLabelPredictionPair> allPairs
    ) {
//        PrecisionRecall<Vector, Integer> precisionRecall = new PrecisionRecall<>(1000);
//        precisionRecall.addResult("Train", trainPredictions, instance -> trueLabels.get(instance.source()) == 1);
//        precisionRecall.addResult("Test", testPredictions, instance -> trueLabels.get(instance.source()) == 1);
//        precisionRecall.addResult("All", allPredictions, instance -> trueLabels.get(instance.source()) == 1);
//        double trainAccuracy = precisionRecall.getAreaUnderCurve("Train");
//        double testAccuracy = precisionRecall.getAreaUnderCurve("Test");
//        double overallAccuracy = precisionRecall.getAreaUnderCurve("All");
        Collections.sort(trainPairs, (p1, p2) -> (int) Math.signum(p2.prediction - p1.prediction));
        Collections.sort(testPairs, (p1, p2) -> (int) Math.signum(p2.prediction - p1.prediction));
        Collections.sort(allPairs, (p1, p2) -> (int) Math.signum(p2.prediction - p1.prediction));
        double trainAUC = PrecisionRecall.areaUnderTheCurve(trainPairs.stream().map(p -> p.trueLabel).collect(Collectors.toList()), trainPairs.stream().map(p -> p.prediction).collect(Collectors.toList()));
        double testAUC = PrecisionRecall.areaUnderTheCurve(testPairs.stream().map(p -> p.trueLabel).collect(Collectors.toList()), testPairs.stream().map(p -> p.prediction).collect(Collectors.toList()));
        double allAUC = PrecisionRecall.areaUnderTheCurve(allPairs.stream().map(p -> p.trueLabel).collect(Collectors.toList()), allPairs.stream().map(p -> p.prediction).collect(Collectors.toList()));
        return new BinaryClassificationEvaluationResults(trainAUC, testAUC, allAUC);
    }

    private EvaluationResults evaluateMultiClassClassificationResults(Map<Integer, Integer> predictions,
                                                                      Map<Integer, Vector> trainingData,
                                                                      int numberOfClasses) {
        double[] true_positives_train = new double[numberOfClasses];
        double[] false_positives_train = new double[numberOfClasses];
        double[] false_negatives_train = new double[numberOfClasses];
        double[] true_positives_test = new double[numberOfClasses];
        double[] false_positives_test = new double[numberOfClasses];
        double[] false_negatives_test = new double[numberOfClasses];
        for (Map.Entry<Integer, Integer> prediction : predictions.entrySet()) {
            int trueLabel = trueLabels.get(prediction.getKey());
            if (trainingData.containsKey(prediction.getKey())) {
                if (prediction.getValue() == trueLabel) {
                    true_positives_train[trueLabel]++;
                } else {
                    false_positives_train[prediction.getValue()]++;
                    false_negatives_train[trueLabel]++;
                }
            } else {
                if (prediction.getValue() == trueLabel) {
                    true_positives_test[trueLabel]++;
                } else {
                    false_positives_test[prediction.getValue()]++;
                    false_negatives_test[trueLabel]++;
                }
            }
        }
        double trainMicroPrecision = computePrecision(ArrayUtilities.sum(true_positives_train), ArrayUtilities.sum(false_positives_train));
        double trainMicroRecall = computeRecall(ArrayUtilities.sum(true_positives_train), ArrayUtilities.sum(false_negatives_train));
        double testMicroPrecision = computePrecision(ArrayUtilities.sum(true_positives_test), ArrayUtilities.sum(false_positives_test));
        double testMicroRecall = computeRecall(ArrayUtilities.sum(true_positives_test), ArrayUtilities.sum(false_negatives_test));
        double overallMicroPrecision = computePrecision(ArrayUtilities.sum(true_positives_train) + ArrayUtilities.sum(true_positives_test),
                                                        ArrayUtilities.sum(false_positives_train) + ArrayUtilities.sum(false_positives_test));
        double overallMicroRecall = computeRecall(ArrayUtilities.sum(true_positives_train) + ArrayUtilities.sum(true_positives_test),
                                                  ArrayUtilities.sum(false_negatives_train) + ArrayUtilities.sum(false_negatives_test));
        double trainMacroF1 = 0.0;
        double testMacroF1 = 0.0;
        double overallMacroF1 = 0.0;
        for (int classIndex = 0; classIndex < numberOfClasses; classIndex++) {
            double trainPrecision = computePrecision(true_positives_train[classIndex], false_positives_train[classIndex]);
            double trainRecall = computeRecall(true_positives_train[classIndex], false_negatives_train[classIndex]);
            trainMacroF1 += computeF1Score(trainPrecision, trainRecall);
            double testPrecision = computePrecision(true_positives_test[classIndex], false_positives_test[classIndex]);
            double testRecall = computeRecall(true_positives_test[classIndex], false_negatives_test[classIndex]);
            testMacroF1 += computeF1Score(testPrecision, testRecall);
            double overallPrecision = computePrecision(true_positives_train[classIndex] + true_positives_test[classIndex],
                                                       false_positives_train[classIndex] + false_positives_test[classIndex]);
            double overallRecall = computeRecall(true_positives_train[classIndex] + true_positives_test[classIndex],
                                                 false_negatives_train[classIndex] + false_negatives_test[classIndex]);
            overallMacroF1 += computeF1Score(overallPrecision, overallRecall);
        }
        trainMacroF1 /= numberOfClasses;
        testMacroF1 /= numberOfClasses;
        overallMacroF1 /= numberOfClasses;
        return new MultiClassClassificationEvaluationResults(
                computeF1Score(trainMicroPrecision, trainMicroRecall),
                computeF1Score(testMicroPrecision, testMicroRecall),
                computeF1Score(overallMicroPrecision, overallMicroRecall),
                trainMacroF1,
                testMacroF1,
                overallMacroF1);
    }

    private static double computePrecision(double true_positives, double false_positives) {
        if (true_positives > 0.0)
            return true_positives / (true_positives + false_positives);
        else if (false_positives > 0.0)
            return 0.0;
        else
            return 1.0;
    }

    private static double computeRecall(double true_positives, double false_negatives) {
        if (true_positives > 0.0)
            return true_positives / (true_positives + false_negatives);
        else if (false_negatives > 0.0)
            return 0.0;
        else
            return 1.0;
    }

    private static double computeF1Score(double precision, double recall) {
        return 2 * precision * recall / (precision + recall);
    }

    private interface EvaluationResults {
        String toString();
        EvaluationResults averageResults(EvaluationResults... results); // TODO: This is kind of a hack to save time. This method should have been static.
    }

    private static class BinaryClassificationEvaluationResults implements EvaluationResults {
        private final double trainAUC;
        private final double testAUC;
        private final double overallAUC;
        private final double trainAUCStandardDeviation;
        private final double testAUCStandardDeviation;
        private final double overallAUCStandardDeviation;

        private BinaryClassificationEvaluationResults(double trainAUC,
                                                      double testAUC,
                                                      double overallAUC) {
            this(trainAUC, testAUC, overallAUC, 0.0, 0.0, 0.0);
        }

        private BinaryClassificationEvaluationResults(double trainAUC,
                                                      double testAUC,
                                                      double overallAUC,
                                                      double trainAUCStandardDeviation,
                                                      double testAUCStandardDeviation,
                                                      double overallAUCStandardDeviation) {
            this.trainAUC = trainAUC;
            this.testAUC = testAUC;
            this.overallAUC = overallAUC;
            this.trainAUCStandardDeviation = trainAUCStandardDeviation;
            this.testAUCStandardDeviation = testAUCStandardDeviation;
            this.overallAUCStandardDeviation = overallAUCStandardDeviation;
        }

        @Override
        public String toString() {
            return trainAUC + "|" + trainAUCStandardDeviation + "\t" +
                    testAUC + "|" + testAUCStandardDeviation + "\t" +
                    overallAUC + "|" + overallAUCStandardDeviation;
        }

        @Override
        public EvaluationResults averageResults(EvaluationResults... results) {
            double[] trainAUCs = new double[results.length];
            double[] testAUCs = new double[results.length];
            double[] overallAUCs = new double[results.length];
            for (int resultIndex = 0; resultIndex < results.length; resultIndex++) {
                trainAUCs[resultIndex] = ((BinaryClassificationEvaluationResults) results[resultIndex]).trainAUC;
                testAUCs[resultIndex] = ((BinaryClassificationEvaluationResults) results[resultIndex]).testAUC;
                overallAUCs[resultIndex] = ((BinaryClassificationEvaluationResults) results[resultIndex]).overallAUC;
            }
            return new BinaryClassificationEvaluationResults(StatisticsUtilities.mean(trainAUCs),
                                                             StatisticsUtilities.mean(testAUCs),
                                                             StatisticsUtilities.mean(overallAUCs),
                                                             StatisticsUtilities.standardDeviation(trainAUCs),
                                                             StatisticsUtilities.standardDeviation(testAUCs),
                                                             StatisticsUtilities.standardDeviation(overallAUCs));
        }
    }

    private static class MultiClassClassificationEvaluationResults implements EvaluationResults {
        private final double trainMicroF1;
        private final double testMicroF1;
        private final double overallMicroF1;
        private final double trainMicroF1StandardDeviation;
        private final double testMicroF1StandardDeviation;
        private final double overallMicroF1StandardDeviation;
        private final double trainMacroF1;
        private final double testMacroF1;
        private final double overallMacroF1;
        private final double trainMacroF1StandardDeviation;
        private final double testMacroF1StandardDeviation;
        private final double overallMacroF1StandardDeviation;

        private MultiClassClassificationEvaluationResults(double trainMicroF1,
                                                          double testMicroF1,
                                                          double overallMicroF1,
                                                          double trainMacroF1,
                                                          double testMacroF1,
                                                          double overallMacroF1) {
            this(trainMicroF1, testMicroF1, overallMicroF1, 0.0, 0.0, 0.0,
                 trainMacroF1, testMacroF1, overallMacroF1, 0.0, 0.0, 0.0);
        }

        private MultiClassClassificationEvaluationResults(double trainMicroF1,
                                                          double testMicroF1,
                                                          double overallMicroF1,
                                                          double trainMicroF1StandardDeviation,
                                                          double testMicroF1StandardDeviation,
                                                          double overallMicroF1StandardDeviation,
                                                          double trainMacroF1,
                                                          double testMacroF1,
                                                          double overallMacroF1,
                                                          double trainMacroF1StandardDeviation,
                                                          double testMacroF1StandardDeviation,
                                                          double overallMacroF1StandardDeviation) {
            this.trainMicroF1 = trainMicroF1;
            this.testMicroF1 = testMicroF1;
            this.overallMicroF1 = overallMicroF1;
            this.trainMicroF1StandardDeviation = trainMicroF1StandardDeviation;
            this.testMicroF1StandardDeviation = testMicroF1StandardDeviation;
            this.overallMicroF1StandardDeviation = overallMicroF1StandardDeviation;
            this.trainMacroF1 = trainMacroF1;
            this.testMacroF1 = testMacroF1;
            this.overallMacroF1 = overallMacroF1;
            this.trainMacroF1StandardDeviation = trainMacroF1StandardDeviation;
            this.testMacroF1StandardDeviation = testMacroF1StandardDeviation;
            this.overallMacroF1StandardDeviation = overallMacroF1StandardDeviation;
        }

        @Override
        public String toString() {
            return trainMicroF1 + "|" + trainMicroF1StandardDeviation + ":" + trainMacroF1 + "|" + trainMacroF1StandardDeviation + "\t" +
                    testMicroF1 + "|" + testMicroF1StandardDeviation + ":" + testMacroF1 + "|" + testMacroF1StandardDeviation + "\t" +
                    overallMicroF1 + "|" + overallMicroF1StandardDeviation + ":" + overallMacroF1 + "|" + overallMacroF1StandardDeviation;
        }

        @Override
        public EvaluationResults averageResults(EvaluationResults... results) {
            double[] trainMicroF1 = new double[results.length];
            double[] testMicroF1 = new double[results.length];
            double[] overallMicroF1 = new double[results.length];
            double[] trainMacroF1 = new double[results.length];
            double[] testMacroF1 = new double[results.length];
            double[] overallMacroF1 = new double[results.length];
            for (int resultIndex = 0; resultIndex < results.length; resultIndex++) {
                trainMicroF1[resultIndex] = ((MultiClassClassificationEvaluationResults) results[resultIndex]).trainMicroF1;
                testMicroF1[resultIndex] = ((MultiClassClassificationEvaluationResults) results[resultIndex]).testMicroF1;
                overallMicroF1[resultIndex] = ((MultiClassClassificationEvaluationResults) results[resultIndex]).overallMicroF1;
                trainMacroF1[resultIndex] = ((MultiClassClassificationEvaluationResults) results[resultIndex]).trainMacroF1;
                testMacroF1[resultIndex] = ((MultiClassClassificationEvaluationResults) results[resultIndex]).testMacroF1;
                overallMacroF1[resultIndex] = ((MultiClassClassificationEvaluationResults) results[resultIndex]).overallMacroF1;
            }
            return new MultiClassClassificationEvaluationResults(StatisticsUtilities.mean(trainMicroF1),
                                                                 StatisticsUtilities.mean(testMicroF1),
                                                                 StatisticsUtilities.mean(overallMicroF1),
                                                                 StatisticsUtilities.standardDeviation(trainMicroF1),
                                                                 StatisticsUtilities.standardDeviation(testMicroF1),
                                                                 StatisticsUtilities.standardDeviation(overallMicroF1),
                                                                 StatisticsUtilities.mean(trainMacroF1),
                                                                 StatisticsUtilities.mean(testMacroF1),
                                                                 StatisticsUtilities.mean(overallMacroF1),
                                                                 StatisticsUtilities.standardDeviation(trainMacroF1),
                                                                 StatisticsUtilities.standardDeviation(testMacroF1),
                                                                 StatisticsUtilities.standardDeviation(overallMacroF1));
        }
    }

    private void appendEvaluationResultsToFile(int featureVectorsSize,
                                               int maximumNumberOfSteps,
                                               EvaluationResults results,
                                               String folderPath) {
        try {
            String resultsLine = featureVectorsSize + "\t" + maximumNumberOfSteps + "\t" + results.toString() + "\n";
            Files.write(Paths.get(folderPath + "/results_" + normalizeDataSetName(dataSetName) + "_" + updateFunctionType.name().toLowerCase() + ".txt"),
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
        DataSets.LabeledDataSet dataSet = DataSets.loadLabeledDataSet(args[5], args[6].equals("1"));
        VertexClassificationExperiment experiment = new VertexClassificationExperiment(dataSet, UpdateFunctionType.valueOf(args[0]), args[7]);
        experiment.runRNNExperiments(intArrayFromString(args[1]), intArrayFromString(args[2]), Integer.parseInt(args[3]), Integer.parseInt(args[4]));
    }

    private static int[] intArrayFromString(String string) {
        String[] stringParts = string.split(",");
        int[] array = new int[stringParts.length];
        for (int partIndex = 0; partIndex < stringParts.length; partIndex++)
            array[partIndex] = Integer.parseInt(stringParts[partIndex]);
        return array;
    }
}
