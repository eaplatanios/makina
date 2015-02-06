package org.platanios.experiment;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.classification.LogisticRegressionAdaGrad;
import org.platanios.learn.classification.TrainableClassifier;
import org.platanios.learn.classification.reflection.Integrator;
import org.platanios.learn.data.*;
import org.platanios.learn.math.matrix.SparseVector;
import org.platanios.learn.math.matrix.Vector;

import java.io.*;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class IntegratorExperiment {
    private static final Logger logger = LogManager.getLogger("Classification / Integrator / Experiment");
    private static final NumberFormat formatter = new DecimalFormat("#0.00");

    private static final Random random = new Random(1000);
    private static final int numberOfIterations = 100;
    private static final int numberOfViews = 3;
    private static final FeatureMapMySQL<SparseVector> featureMap = new FeatureMapMySQL<>(
            numberOfViews,
            "jdbc:mysql://rtw.ml.cmu.edu/",
            "eplatani",
            "Ant0nisS719791!",
            "learn",
            "features"
    );
    private static final String filteredLabeledDataDirectory = "/home/eplatani/Integrator Experiment/Training Data/filtered_labeled_nps.data";
    private static final String integratorWorkingDirectory = "/home/eplatani/Integrator Experiment/Integrator Working Directory/";
    private static final Map<String, Map<String, Boolean>> filteredLabeledData = FeaturesPreprocessing.readStringStringBooleanMap(filteredLabeledDataDirectory);
    private static final String category = "animal";
    private static final List<String> seeds = Arrays.asList(
            "cattle", "livestock", "pigs", "amazons", "badgers", "barbs", "bluegills", "bowhead whales", "brontosauruses",
            "bryozoans", "bunnies", "bunnies", "carnivores", "cavity nesters", "cavity nesters", "cetaceans", "cnidarians",
            "corvids", "cow", "creatures", "cyprinids", "dog", "echinoderms", "endangered species", "filter feeders",
            "furbearers", "great apes", "guppies", "hammerheads", "herbor seals", "hawks", "herbivores", "humans", "hyenas",
            "insectivores", "invertebrates", "leatherjackets", "lemurs", "locusts", "manatees", "marine animals", "marine life",
            "marine mammals", "marlins", "minnows", "monkey", "cat", "fish"
    );
    private static final int evaluationDataSetSize = 200;

    private static MultiViewDataSet<MultiViewPredictedDataInstance<Vector, Double>> labeledDataSet;
    private static MultiViewDataSet<MultiViewPredictedDataInstance<Vector, Double>> unlabeledDataSet;
    private static MultiViewDataSet<MultiViewPredictedDataInstance<Vector, Double>> evaluationDataSet;

    private static void initializeWorkingDirectory(String workingDirectory) {
        File directory = new File(workingDirectory);
        if (!directory.exists() && !directory.mkdirs())
            logger.error("Unable to create directory " + directory.getAbsolutePath());
        directory = new File(workingDirectory + "MATLAB Plotting Scripts/");
        if (!directory.exists() && !directory.mkdirs())
            logger.error("Unable to create directory " + directory.getAbsolutePath());
    }

    private static void loadLabeledNPsData() {
        labeledDataSet = new MultiViewDataSetInMemory<>();
        unlabeledDataSet = new MultiViewDataSetInMemory<>();
        evaluationDataSet = new MultiViewDataSetInMemory<>();
        for (String seed : seeds) {
            if (filteredLabeledData.get(category).getOrDefault(seed, null) != null) {
                labeledDataSet.add(
                        new MultiViewPredictedDataInstance<>(seed,
                                                             featureMap.getFeatureVectors(seed).stream().collect(Collectors.toList()),
                                                             filteredLabeledData.get(category).get(seed) ? 1.0 : 0.0,
                                                             null,
                                                             1)
                );
            }
        }
        List<Map.Entry<String, Boolean>> filteredLabeledDataNPs = new ArrayList<>(filteredLabeledData.get(category).entrySet());
        Collections.shuffle(filteredLabeledDataNPs, random);
        int numberOfEntriesProcessed = 0;
        int numberOfPositiveLabeledDataSamples = labeledDataSet.size();
        int numberOfNegativeLabeledDataSamples = 0;
        for (Map.Entry<String, Boolean> dataSample : filteredLabeledDataNPs) {
            if (!dataSample.getValue() && numberOfNegativeLabeledDataSamples < numberOfPositiveLabeledDataSamples) {
                String np = dataSample.getKey();
                labeledDataSet.add(
                        new MultiViewPredictedDataInstance<>(np,
                                                             featureMap.getFeatureVectors(np).stream().collect(Collectors.toList()),
                                                             dataSample.getValue() ? 1.0 : 0.0,
                                                             null,
                                                             1)
                );
                numberOfNegativeLabeledDataSamples++;
            }
            if (numberOfEntriesProcessed <= filteredLabeledData.get(category).size() - evaluationDataSetSize) {
                String np = dataSample.getKey();
                if (!seeds.contains(np)) {
                    unlabeledDataSet.add(
                            new MultiViewPredictedDataInstance<>(np,
                                                                 featureMap.getFeatureVectors(np).stream().collect(Collectors.toList()),
                                                                 dataSample.getValue() ? 1.0 : 0.0,
                                                                 null,
                                                                 1)
                    );
                }
            } else {
                String np = dataSample.getKey();
                if (!seeds.contains(np)) {
                    evaluationDataSet.add(
                            new MultiViewPredictedDataInstance<>(np,
                                                                 featureMap.getFeatureVectors(np).stream().collect(Collectors.toList()),
                                                                 dataSample.getValue() ? 1.0 : 0.0,
                                                                 null,
                                                                 1)
                    );
                }
            }
            numberOfEntriesProcessed++;
        }
    }

    @SuppressWarnings("unchecked")
    public static void main(String[] args) {
        Integrator.CoTrainingMethod coTrainingMethod;
        String workingDirectory;
        switch (args[0]) {
            case "0":
                coTrainingMethod = Integrator.CoTrainingMethod.CO_TRAINING;
                workingDirectory = integratorWorkingDirectory + "Co-Training/";
                break;
            case "1":
                coTrainingMethod = Integrator.CoTrainingMethod.ROBUST_CO_TRAINING;
                workingDirectory = integratorWorkingDirectory + "Robust Co-Training/";
                break;
            case "2":
                coTrainingMethod = Integrator.CoTrainingMethod.ROBUST_CO_TRAINING_GM;
                workingDirectory = integratorWorkingDirectory + "Robust Co-Training GM/";
                break;
            default:
                coTrainingMethod = Integrator.CoTrainingMethod.CO_TRAINING;
                workingDirectory = integratorWorkingDirectory + "Co-Training/";
        }
        initializeWorkingDirectory(workingDirectory);
        loadLabeledNPsData();
        double[] weightedMajorityAccuracy = new double[numberOfIterations];
        double[][] actualErrorRates = new double[numberOfIterations][numberOfViews];
        double[][] precision = new double[numberOfIterations][numberOfViews];
        double[][] recall = new double[numberOfIterations][numberOfViews];
        double[][] f1Score = new double[numberOfIterations][numberOfViews];
        double[][] estimatedErrorRates = new double[numberOfIterations][numberOfViews];
        Integrator.Builder<Vector, Double> integratorBuilder =
                new Integrator.Builder<Vector, Double>(integratorWorkingDirectory)
                        .labeledDataSet(labeledDataSet)
                        .unlabeledDataSet(unlabeledDataSet)
                        .completedIterationEventHandlers((completedIterationEvent, sequence, endOfBatch) -> {
                            int iterationNumber = ((Integrator.CompletedIterationEvent) completedIterationEvent).getIterationNumber();
                            List<TrainableClassifier<Vector, Double>> classifiers = ((Integrator.CompletedIterationEvent) completedIterationEvent).getClassifiers();
                            estimatedErrorRates[iterationNumber] = ((Integrator.CompletedIterationEvent) completedIterationEvent).getErrorRates();
                            for (int view = 0; view < numberOfViews; view++) {
                                DataSet<PredictedDataInstance<Vector, Double>> predictedEvaluationDataSet =
                                        classifiers.get(view).predict((DataSet<PredictedDataInstance<Vector, Double>>) evaluationDataSet.getSingleViewDataSet(view));
                                double precisionDenominator = 0;
                                double recallDenominator = 0;
                                for (PredictedDataInstance<Vector, Double> dataInstance : predictedEvaluationDataSet) {
                                    if (dataInstance.label() != (filteredLabeledData.get(category).get(dataInstance.name()) ? 1 : 0))
                                        actualErrorRates[iterationNumber][view]++;
                                    if (dataInstance.label() == 1) {
                                        precisionDenominator++;
                                        if (filteredLabeledData.get(category).get(dataInstance.name()))
                                            precision[iterationNumber][view]++;
                                    }
                                    if (filteredLabeledData.get(category).get(dataInstance.name())) {
                                        recallDenominator++;
                                        if (dataInstance.label() == 1)
                                            recall[iterationNumber][view]++;
                                    }
                                }
                                if (precisionDenominator == 0)
                                    precisionDenominator = 1;
                                if (recallDenominator == 0)
                                    recallDenominator = 1;
                                actualErrorRates[iterationNumber][view] /= predictedEvaluationDataSet.size();
                                precision[iterationNumber][view] /= precisionDenominator;
                                recall[iterationNumber][view] /= recallDenominator;
                                f1Score[iterationNumber][view] = 2
                                        * (precision[iterationNumber][view] * recall[iterationNumber][view])
                                        / (precision[iterationNumber][view] + recall[iterationNumber][view]);
                                if (precision[iterationNumber][view] == 0 && recall[iterationNumber][view] == 0)
                                    f1Score[iterationNumber][view] = 0;
                                logger.info("Iteration #" + (iterationNumber + 1) + ", Classifier #" + (view + 1) + " | " +
                                                    "Actual Error Rate: " + formatter.format(actualErrorRates[iterationNumber][view]) + " | " +
                                                    "Estimated Error Rate: " + formatter.format(estimatedErrorRates[iterationNumber][view]) + " | " +
                                                    "Precision: " + formatter.format(precision[iterationNumber][view]) + " | " +
                                                    "Recall: " + formatter.format(recall[iterationNumber][view]) + " | " +
                                                    "F-1 Score: " + formatter.format(f1Score[iterationNumber][view]) + " |"
                                );
                            }
                            saveResultsForMATLAB(workingDirectory + "MATLAB Plotting Scripts/",
                                                 actualErrorRates,
                                                 estimatedErrorRates,
                                                 precision,
                                                 recall,
                                                 f1Score);
                        })
                        .coTrainingMethod(Integrator.CoTrainingMethod.ROBUST_CO_TRAINING_GM)
                        .dataSelectionMethod(Integrator.DataSelectionMethod.FIXED_PROPORTION)
                        .dataSelectionParameter(0.01)
                        .numberOfThreads(numberOfViews)
                        .saveModelsOnEveryIteration(true)
                        .useDifferentFilePerIteration(true);
        for (int view = 0; view < numberOfViews; view++) {
            LogisticRegressionAdaGrad classifier =
                    new LogisticRegressionAdaGrad.Builder(unlabeledDataSet
                                                                  .get(0)
                                                                  .getSingleViewDataInstance(view)
                                                                  .features()
                                                                  .size())
                            .sparse(true)
                            .useBiasTerm(true)
                            .useL1Regularization(false)
                            .l1RegularizationWeight(0.0001)
                            .useL2Regularization(false)
                            .l2RegularizationWeight(0.1)
                            .loggingLevel(0)
                            .sampleWithReplacement(true)
                            .maximumNumberOfIterations(1000)
                            .maximumNumberOfIterationsWithNoPointChange(10)
                            .pointChangeTolerance(1e-5)
                            .checkForPointConvergence(true)
                            .batchSize(10000000)
                            .random(random)
                            .build();
            integratorBuilder.addClassifier(classifier);
        }
        Integrator<Vector, Double> integrator = integratorBuilder.build();
        for (int iteration = integrator.getIterationNumber(); iteration < numberOfIterations; iteration++)
            integrator.performSingleIteration();
    }

    private static void saveResultsForMATLAB(
            String directory,
            double[][] actualErrorRates,
            double[][] estimatedErrorRates,
            double[][] precision,
            double[][] recall,
            double[][] f1Score
    ) {
        File outputFile = new File(directory + File.separator + "data.m");

        try {
            if (!outputFile.exists() && !outputFile.createNewFile())
                logger.error("Could not create the file \"" + outputFile.getAbsolutePath() + "\" to store the MATLAB scripts!");
            FileWriter fileWriter = new FileWriter(outputFile, false);
            fileWriter.append("actual_error_rates = " + convertToMatlabMatrixString(actualErrorRates) + "\n");
            fileWriter.append("estimated_error_rates = " + convertToMatlabMatrixString(estimatedErrorRates) + "\n");
            fileWriter.append("precision = " + convertToMatlabMatrixString(precision) + "\n");
            fileWriter.append("recall = " + convertToMatlabMatrixString(recall) + "\n");
            fileWriter.append("f1_score = " + convertToMatlabMatrixString(f1Score) + "\n");
            fileWriter.close();
        } catch (IOException e) {
            logger.error("Could not create or open the file \"" + outputFile.getAbsolutePath() + "\" to store the MATLAB scripts!");
        }
    }

    private static String convertToMatlabMatrixString(double[][] array) {
        String string = "[ ";
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++)
                string += array[i][j] + ", ";
            string = string.substring(0, string.length() - 2) + "; ";
        }
        return string.substring(0, string.length() - 2) + "];";
    }
}
