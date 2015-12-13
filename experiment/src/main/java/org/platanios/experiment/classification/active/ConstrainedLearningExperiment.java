package org.platanios.experiment.classification.active;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.experiment.Utilities;
import org.platanios.learn.classification.LogisticRegressionAdaGrad;
import org.platanios.learn.classification.active.*;
import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.DataSetInMemory;
import org.platanios.learn.data.LabeledDataInstance;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.evaluation.PrecisionRecall;
import org.platanios.learn.math.matrix.SparseVector;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorType;
import org.platanios.learn.math.matrix.Vectors;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ConstrainedLearningExperiment {
    private static final Logger logger = LogManager.getLogger("Classification / Active / Constrained Learning Experiment");

    private final int initialNumberOfExamples;
    private final double initialRatioOfPositiveExamples;
    private final int numberOfExamplesToPickPerIteration;
    private final ActiveLearningMethod activeLearningMethod;
    private final Set<Label> labels;
    private final Constraint constraint;
    private final Map<String, Vector> featureMap;

    private final Map<Label, DataSetStatistics> dataSetStatistics = new HashMap<>();
    private final Map<Label, DataSet<LabeledDataInstance<Vector, Double>>> labeledDataSet = new HashMap<>();
    private final Map<Label, Map<String, Boolean>> trueLabels = new HashMap<>();
    private final Map<String, Map<Label, Boolean>> fixedLabels = new HashMap<>();

    private ConstrainedLearningExperiment(int initialNumberOfExamples,
                                          double initialRatioOfPositiveToNegativeExamples,
                                          int numberOfExamplesToPickPerIteration,
                                          ActiveLearningMethod activeLearningMethod,
                                          String cplFeatureMapDirectory,
                                          String workingDirectory) {
        this.initialNumberOfExamples = initialNumberOfExamples;
        this.initialRatioOfPositiveExamples = initialRatioOfPositiveToNegativeExamples;
        this.numberOfExamplesToPickPerIteration = numberOfExamplesToPickPerIteration;
        this.activeLearningMethod = activeLearningMethod;
        logger.info("Importing labeled noun phrases...");
        Map<String, Set<String>> labeledNounPhrases = importLabeledNounPhrases(workingDirectory);
        labels = labeledNounPhrases.values().stream().flatMap(Collection::stream).map(Label::new).collect(Collectors.toSet());
        logger.info("Importing constraints...");
        constraint = importConstraints(workingDirectory);
        logger.info("Importing feature map...");
        if (Files.exists(Paths.get(workingDirectory + "/features.bin")))
            featureMap = Utilities.readMap(workingDirectory + "/features.bin");
        else
            featureMap = buildFeatureMap(workingDirectory + "/features.bin",
                                         cplFeatureMapDirectory,
                                         labeledNounPhrases.keySet());
        for (Label label : labels) {
            dataSetStatistics.put(label, new DataSetStatistics());
            labeledDataSet.put(label, new DataSetInMemory<>());
            trueLabels.put(label, new HashMap<>());
        }
        Set<String> nounPhrasesWithoutFeatures = new HashSet<>();
        Map<String, Set<String>> filteredLabeledNounPhrases = new HashMap<>();
        for (Map.Entry<String, Set<String>> labeledNounPhraseEntry : labeledNounPhrases.entrySet()) {
            String nounPhrase = labeledNounPhraseEntry.getKey();
            Set<String> positiveLabels = labeledNounPhraseEntry.getValue();
            Set<String> negativeLabels = labels.stream().map(Label::getName).collect(Collectors.toSet());
            negativeLabels.removeAll(positiveLabels);
            Vector features;
            if (!featureMap.containsKey(nounPhrase)) {
                nounPhrasesWithoutFeatures.add(nounPhrase);
                for (String labelName : positiveLabels)
                    dataSetStatistics.get(new Label(labelName)).numberOfPositiveExamplesWithoutFeatures++;
                continue;
            } else {
                filteredLabeledNounPhrases.put(nounPhrase, positiveLabels);
                features = featureMap.get(nounPhrase);
                fixedLabels.put(nounPhrase, new HashMap<>());
            }
            for (String labelName : positiveLabels) {
                Label label = new Label(labelName);
                dataSetStatistics.get(label).numberOfPositiveExamples++;
                dataSetStatistics.get(label).totalNumberOfExamples++;
                labeledDataSet.get(label).add(new LabeledDataInstance<>(nounPhrase, features, 1.0, null));
                trueLabels.get(label).put(nounPhrase, true);
            }
            for (String labelName : negativeLabels) {
                Label label = new Label(labelName);
                dataSetStatistics.get(label).numberOfNegativeExamples++;
                dataSetStatistics.get(label).totalNumberOfExamples++;
                labeledDataSet.get(label).add(new LabeledDataInstance<>(nounPhrase, features, 0.0, null));
                trueLabels.get(label).put(nounPhrase, false);
            }
        }
        logger.info("Noun phrases without features that were ignored: " + nounPhrasesWithoutFeatures);
        exportLabeledNounPhrases(filteredLabeledNounPhrases, workingDirectory + "/filtered_labeled_nps.tsv");
    }

    private ExperimentResults runExperiment() {
        logger.info("Running experiment...");
        final long experimentStartTime = System.currentTimeMillis();
        ExperimentResults results = new ExperimentResults();
        Map<Label, DataSet<LabeledDataInstance<Vector, Double>>> trainingDataSet = new HashMap<>();
        Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> testingDataSet = new HashMap<>();
        ConstrainedLearning.Builder<Vector> learningBuilder =
                new ConstrainedLearning.Builder<>()
                        .activeLearningMethod(activeLearningMethod)
                        .addConstraint(constraint);
        int numberOfUnlabeledExamples = 0;
        for (Label label : labels) {
            trainingDataSet.put(label, new DataSetInMemory<>());
            testingDataSet.put(label, new DataSetInMemory<>());
            labeledDataSet.get(label).shuffle();
            int numberOfPositiveExamplesNeeded =
                    (int) Math.ceil(initialRatioOfPositiveExamples * initialNumberOfExamples);
            int numberOfPositiveExamplesAdded = 0;
            int numberOfNegativeExamplesAdded = 0;
            for (LabeledDataInstance<Vector, Double> dataInstance : labeledDataSet.get(label)) {
                if (dataInstance.label() >= 0.5 && numberOfPositiveExamplesAdded < numberOfPositiveExamplesNeeded) {
                    trainingDataSet.get(label).add(dataInstance);
                    fixedLabels.get(dataInstance.name()).put(label, true);
                    numberOfPositiveExamplesAdded++;
                } else if (dataInstance.label() < 0.5
                        && numberOfNegativeExamplesAdded < initialNumberOfExamples - numberOfPositiveExamplesNeeded) {
                    trainingDataSet.get(label).add(dataInstance);
                    fixedLabels.get(dataInstance.name()).put(label, false);
                    numberOfNegativeExamplesAdded++;
                } else {
                    testingDataSet.get(label).add(new PredictedDataInstance<>(dataInstance.name(),
                                                                              dataInstance.features(),
                                                                              dataInstance.label(),
                                                                              dataInstance.source(),
                                                                              0.5));
                    numberOfUnlabeledExamples++;
                }
            }
            learningBuilder.addLabel(label,
                                     new LogisticRegressionAdaGrad.Builder(trainingDataSet.get(label).get(0).features().size())
                                             .sparse(true)
                                             .useBiasTerm(true)
//                                             .useL1Regularization(true)
//                                             .l1RegularizationWeight(0.01)
                                             .useL2Regularization(true)
                                             .l2RegularizationWeight(0.01)
                                             .loggingLevel(0)
                                             .sampleWithReplacement(true)
                                             .maximumNumberOfIterations(1000)
                                             .maximumNumberOfIterationsWithNoPointChange(10)
                                             .pointChangeTolerance(1e-5)
                                             .checkForPointConvergence(true)
                                             .batchSize(1000)
                                             .build());
        }
        numberOfUnlabeledExamples -= propagateConstraints(trainingDataSet, testingDataSet);
        Learning<Vector> learning = learningBuilder.build();
        int numberOfExamplesPicked = 0;
        int iterationNumber = 0;
        while (true) {
            PrecisionRecall<Vector, Double> precisionRecall = new PrecisionRecall<>(1000);
            PrecisionRecall<Vector, Double> testingPrecisionRecall = new PrecisionRecall<>(1000);
            labels.parallelStream().forEach(label -> {
                learning.trainClassifier(label, trainingDataSet.get(label));
                learning.makePredictions(label, testingDataSet.get(label));
                DataSet<PredictedDataInstance<Vector, Double>> evaluationDataSet = new DataSetInMemory<>();
                for (LabeledDataInstance<Vector, Double> dataInstance : trainingDataSet.get(label))
                    evaluationDataSet.add(new PredictedDataInstance<>(
                            dataInstance.name(),
                            dataInstance.features(),
                            1.0,
                            dataInstance.source(),
                            dataInstance.label()
                    ));
                for (PredictedDataInstance<Vector, Double> dataInstance : testingDataSet.get(label))
                    evaluationDataSet.add(dataInstance);
                precisionRecall.addResult(label.getName(),
                                          evaluationDataSet,
                                          dataInstance -> trueLabels.get(label).get(dataInstance.name()));
                testingPrecisionRecall.addResult(label.getName(),
                                                 testingDataSet.get(label),
                                                 dataInstance -> trueLabels.get(label).get(dataInstance.name()));
//                StringJoiner predictedStringJoiner = new StringJoiner(",", "[", "]");
//                StringJoiner targetStringJoiner = new StringJoiner(",", "[", "]");
//                for (PredictedDataInstance<Vector, Double> dataInstance : evaluationDataSet) {
//                    predictedStringJoiner.add("" + dataInstance.probability());
//                    targetStringJoiner.add(trueLabels.get(label).get(dataInstance.name()) ? "1" : "0");
//                }
//                logger.info("PR " + label.getName() + " predicted:\t" + predictedStringJoiner.toString());
//                logger.info("PR " + label.getName() + " target:\t" + targetStringJoiner.toString());
            });
            results.averageAreasUnderTheCurve.put(iterationNumber,
                                                  precisionRecall.getAreaUnderCurves()
                                                          .stream()
                                                          .mapToDouble(area -> area)
                                                          .average()
                                                          .orElse(0));
            results.averageTestingAreasUnderTheCurve.put(iterationNumber,
                                                         testingPrecisionRecall.getAreaUnderCurves()
                                                                 .stream()
                                                                 .mapToDouble(area -> area)
                                                                 .average()
                                                                 .orElse(0));
            results.numberOfExamplesPicked.put(iterationNumber, numberOfExamplesPicked);
            final long startTime = System.currentTimeMillis();
            List<Learning.InstanceToLabel<Vector>> selectedInstances =
                    learning.pickInstancesToLabel(testingDataSet, numberOfExamplesToPickPerIteration);
            final long endTime = System.currentTimeMillis();
            results.activeLearningMethodTimesTaken.put(iterationNumber, endTime - startTime);

            for (Learning.InstanceToLabel<Vector> selectedInstance : selectedInstances) {
                Label label = selectedInstance.getLabel();
                PredictedDataInstance<Vector, Double> selectedDataInstance =
                        testingDataSet.get(label).remove(selectedInstance.getInstance());
                boolean trueLabel = trueLabels.get(label).get(selectedDataInstance.name());
                selectedDataInstance.label(trueLabel ? 1.0 : 0.0);
                trainingDataSet.get(label).add(selectedDataInstance);
                fixedLabels.get(selectedDataInstance.name()).put(label, trueLabel);
            }
            numberOfUnlabeledExamples -= propagateConstraints(trainingDataSet, testingDataSet);
            if (numberOfUnlabeledExamples - numberOfExamplesPicked <= 0)
                break;
            numberOfExamplesPicked += selectedInstances.size();
            iterationNumber++;
        }
        final long experimentEndTime = System.currentTimeMillis();
        results.timeTaken = experimentEndTime - experimentStartTime;
        return results;
    }

    // TODO: Make this method more efficient.
    private int propagateConstraints(Map<Label, DataSet<LabeledDataInstance<Vector, Double>>> trainingDataSet,
                                     Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> testingDataSet) {
        int numberOfLabelsFixed = 0;
        for (Map.Entry<String, Map<Label, Boolean>> fixedLabelsEntry : fixedLabels.entrySet()) {
            if (constraint.propagate(fixedLabelsEntry.getValue()) > 0)
                for (Map.Entry<Label, Boolean> fixedLabelEntry : fixedLabelsEntry.getValue().entrySet()) {
                    for (PredictedDataInstance<Vector, Double> dataInstance : testingDataSet.get(fixedLabelEntry.getKey())) {
                        if (dataInstance.name().equals(fixedLabelsEntry.getKey())) {
                            testingDataSet.get(fixedLabelEntry.getKey()).remove(dataInstance);
                            dataInstance.label(trueLabels.get(fixedLabelEntry.getKey()).get(dataInstance.name()) ? 1.0 : 0.0);
                            trainingDataSet.get(fixedLabelEntry.getKey()).add(dataInstance);
                            numberOfLabelsFixed++;
                        }
                    }
                }
        }
        return numberOfLabelsFixed;
    }

    private Map<String, Vector> buildFeatureMap(String featureMapDirectory, String cplFeatureMapDirectory) {
        return buildFeatureMap(featureMapDirectory, cplFeatureMapDirectory, null);
    }

    private Map<String, Vector> buildFeatureMap(String featureMapDirectory,
                                                String cplFeatureMapDirectory,
                                                Set<String> nounPhrases) {
        Map<String, Vector> featureMap = new HashMap<>();
        Map<String, Integer> contexts;
        try {
            if (Files.exists(Paths.get(cplFeatureMapDirectory + "/contexts.bin")))
                contexts = Utilities.readMap(cplFeatureMapDirectory + "/contexts.bin");
            else
                contexts = buildContextsMap(cplFeatureMapDirectory);
            Stream<String> npContextPairsLines = new BufferedReader(new InputStreamReader(new GZIPInputStream(
                    Files.newInputStream(Paths.get(cplFeatureMapDirectory + "/cat_pairs_np-idx.txt.gz"))
            ))).lines();
            npContextPairsLines.forEachOrdered(line -> {
                String[] lineParts = line.split("\t");
                String np = lineParts[0];
                if (nounPhrases == null || nounPhrases.contains(np)) {
                    SparseVector features = (SparseVector) Vectors.build(contexts.size(), VectorType.SPARSE);
                    for (int i = 1; i < lineParts.length; i++) {
                        String[] contextParts = lineParts[i].split(" -#- ");
                        if (contexts.containsKey(contextParts[0]))
                            features.set(contexts.get(contextParts[0]), Double.parseDouble(contextParts[1]));
                        else
                            System.out.println("error error");
                    }
                    featureMap.put(np, features);
                }
            });
        } catch (IOException e) {
            System.out.println("An exception was thrown while trying to build the CPL feature map.");
        }
//        Utilities.writeMap(featureMap, featureMapDirectory); // TODO: Fix this.
        return featureMap;
    }

    private Map<String, Integer> buildContextsMap(String cplFeatureMapDirectory) throws IOException {
        Map<String, Integer> contexts = new HashMap<>();
        Stream<String> npContextPairsLines = new BufferedReader(new InputStreamReader(new GZIPInputStream(
                Files.newInputStream(Paths.get(cplFeatureMapDirectory + "/cat_contexts.txt.gz"))
        ))).lines();
        int[] contextIndex = {0};
        npContextPairsLines.forEachOrdered(line -> {
            String[] lineParts = line.split("\t");
            if (!contexts.containsKey(lineParts[0]))
                contexts.put(lineParts[0], contextIndex[0]++);
        });
        Utilities.writeMap(contexts, cplFeatureMapDirectory + "/contexts.bin");
        return contexts;
    }

    private void exportLabeledNounPhrases(Map<String, Set<String>> labeledNounPhrases, String filePath) {
        try {
            FileWriter writer = new FileWriter(filePath);
            for (String nounPhrase : labeledNounPhrases.keySet()) {
                StringJoiner stringJoiner = new StringJoiner(",");
                labeledNounPhrases.get(nounPhrase).forEach(stringJoiner::add);
                writer.write(nounPhrase + "\t" + stringJoiner.toString() + "\n");
            }
            writer.close();
        } catch (IOException e) {
            System.out.println("An exception was thrown while trying to export a set of labeled noun phrases.");
            e.printStackTrace();
        }
    }

    private Map<String, Set<String>> importLabeledNounPhrases(String workingDirectory) {
        Map<String, Set<String>> labeledNounPhrases = new HashMap<>();
        try {
            Files.newBufferedReader(Paths.get(workingDirectory + "/labeled_nps.tsv")).lines().forEach(line -> {
                String[] lineParts = line.split("\t");
                if (lineParts.length == 2)
                    labeledNounPhrases.put(lineParts[0], new HashSet<>(Arrays.asList(lineParts[1].split(","))));
            });
        } catch (IOException e) {
            throw new IllegalArgumentException("There was a problem with the provided labeled noun phrases file.");
        }
        return labeledNounPhrases;
    }

    private Constraint importConstraints(String workingDirectory) {
        Set<Constraint> constraints = new HashSet<>();
        try {
            Files.newBufferedReader(Paths.get(workingDirectory + "/constraints.txt")).lines().forEach(line -> {
                if (line.startsWith("!"))
                    constraints.add(new MutualExclusionConstraint(Arrays.asList(line.substring(1).split(","))
                                                                          .stream()
                                                                          .map(Label::new)
                                                                          .collect(Collectors.toSet())));
            });
        } catch (IOException e) {
            throw new IllegalArgumentException("There was a problem with the provided labeled noun phrases file.");
        }
        return new ConstraintSet(constraints);
    }

    private void logDataSetStatistics() {
        StringBuilder stringBuilder = new StringBuilder("Logging data set statistics...\n");
        for (Label label : labels)
            stringBuilder.append("\t").append(label.getName())
                    .append(":\t\t{ ").append(dataSetStatistics.get(label).toString()).append(" }\n");
        logger.info(stringBuilder.toString());
    }

    private static void exportResults(Map<ActiveLearningMethod, List<ExperimentResults>> results, String filePath) {
        try {
            FileWriter writer = new FileWriter(filePath);
            for (Map.Entry<ActiveLearningMethod, List<ExperimentResults>> resultsEntry : results.entrySet()) {
                writer.write(resultsEntry.getKey().name() + "\n");
                int experimentIndex = 1;
                for (ExperimentResults result : resultsEntry.getValue()) {
                    writer.write("\tExperiment " + experimentIndex++ + ":\n");
                    writer.write("\t\tAverage AUC: " + mapToString(result.averageAreasUnderTheCurve) + "\n");
                    writer.write("\t\tAverage Testing AUC: " + mapToString(result.averageTestingAreasUnderTheCurve) + "\n");
                    writer.write("\t\tActive Learning Times: " + mapToString(result.activeLearningMethodTimesTaken) + "\n");
                    writer.write("\t\tTotal time taken: " + result.timeTaken + "\n");
                }
            }
            writer.close();
        } catch (IOException e) {
            System.out.println("An exception was thrown while trying to export a set of experiment results.");
            e.printStackTrace();
        }
    }

    private static String mapToString(Map<Integer, ?> map) {
        StringJoiner indexesStringJoiner = new StringJoiner(",", "[", "]");
        StringJoiner valuesStringJoiner = new StringJoiner(",", "[", "]");
        map.entrySet()
                .stream()
                .sorted(Map.Entry.comparingByKey(Integer::compareTo))
                .forEachOrdered(entry -> {
                    indexesStringJoiner.add("" + entry.getKey());
                    valuesStringJoiner.add("" + entry.getValue());
                });
        return "{ Indexes = " + indexesStringJoiner.toString() + "; " +
                "{ Values = " + valuesStringJoiner.toString() + "; }";
    }

    private static class DataSetStatistics {
        private int numberOfPositiveExamples = 0;
        private int numberOfNegativeExamples = 0;
        private int totalNumberOfExamples = 0;
        private int numberOfPositiveExamplesWithoutFeatures;

        @Override
        public String toString() {
            return "Positive: " + numberOfPositiveExamples
                    + "\tNegative: " + numberOfNegativeExamples
                    + "\tTotal: " + totalNumberOfExamples
                    + "\tWithout Features: " + numberOfPositiveExamplesWithoutFeatures;
        }
    }

    private static class ExperimentResults {
        private Map<Integer, Double> averageAreasUnderTheCurve = new HashMap<>();
        private Map<Integer, Double> averageTestingAreasUnderTheCurve = new HashMap<>();
        private Map<Integer, Integer> numberOfExamplesPicked = new HashMap<>();
        private Map<Integer, Long> activeLearningMethodTimesTaken = new HashMap<>();
        private long timeTaken;
    }

    public static void main(String[] args) {
        int numberOfExperimentRepetitions = 2;
        int initialNumberOfExamples = 10;
        double initialRatioOfPositiveExamples = 0.5;
        int numberOfExamplesToPickPerIteration = 10;
        String cplFeatureMapDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/Data Sets/NELL/Server/all-pairs/all-pairs-OC-2010-12-01-small200-gz";
        String workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment 0";
        ActiveLearningMethod[] activeLearningMethods =
                new ActiveLearningMethod[] { ActiveLearningMethod.RANDOM, ActiveLearningMethod.UNCERTAINTY_HEURISTIC };
        Map<ActiveLearningMethod, List<ExperimentResults>> results = new HashMap<>();
        for (ActiveLearningMethod activeLearningMethod : activeLearningMethods) {
            logger.info("Running experiments for " + activeLearningMethod.name() + "...");
            results.put(activeLearningMethod, new ArrayList<>());
            ConstrainedLearningExperiment experiment = new ConstrainedLearningExperiment(
                    initialNumberOfExamples,
                    initialRatioOfPositiveExamples,
                    numberOfExamplesToPickPerIteration,
                    activeLearningMethod,
                    cplFeatureMapDirectory,
                    workingDirectory
            );
            experiment.logDataSetStatistics();
            Map<Integer, Double> averageAreasUnderTheCurve = new HashMap<>();
            Map<Integer, Integer> countAreasUnderTheCurve = new HashMap<>();
            for (int experimentRepetition = 0; experimentRepetition < numberOfExperimentRepetitions; experimentRepetition++) {
                logger.info("Running experiment repetition " + (experimentRepetition + 1) + "...");
                ExperimentResults experimentResults = experiment.runExperiment();
                results.get(activeLearningMethod).add(experimentResults);
                for (Map.Entry<Integer, Double> aucEntry : experimentResults.averageAreasUnderTheCurve.entrySet()) {
                    int key = aucEntry.getKey();
                    if (averageAreasUnderTheCurve.containsKey(key)) {
                        averageAreasUnderTheCurve.put(
                                key,
                                averageAreasUnderTheCurve.get(key) + aucEntry.getValue()
                        );
                        countAreasUnderTheCurve.put(key, countAreasUnderTheCurve.get(key) + 1);
                    } else {
                        averageAreasUnderTheCurve.put(key, aucEntry.getValue());
                        countAreasUnderTheCurve.put(key, 1);
                    }
                }
            }
            for (int key : averageAreasUnderTheCurve.keySet())
                averageAreasUnderTheCurve.put(key, averageAreasUnderTheCurve.get(key) / countAreasUnderTheCurve.get(key));
        }
        ConstrainedLearningExperiment.exportResults(results, workingDirectory + "/results.txt");
        logger.info("Finished!");
    }
}
