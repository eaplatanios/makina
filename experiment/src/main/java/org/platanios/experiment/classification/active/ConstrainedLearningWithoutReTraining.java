package org.platanios.experiment.classification.active;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.experiment.Utilities;
import org.platanios.learn.classification.LogisticRegressionAdaGrad;
import org.platanios.learn.classification.TrainableClassifier;
import org.platanios.learn.classification.active.*;
import org.platanios.learn.data.DataInstance;
import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.DataSetInMemory;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.evaluation.PrecisionRecall;
import org.platanios.learn.math.matrix.*;
import org.platanios.learn.math.matrix.Vector;

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
public class ConstrainedLearningWithoutReTraining {
    private static final Logger logger = LogManager.getLogger("Classification / Active / Constrained Learning Experiment");

    private static final Map<ActiveLearningMethod, String> matlabPlotColorsMap = new HashMap<>();
    static {
        matlabPlotColorsMap.put(ActiveLearningMethod.RANDOM, "[0, 0.4470, 0.7410]");
        matlabPlotColorsMap.put(ActiveLearningMethod.UNCERTAINTY_HEURISTIC, "[0.8500, 0.3250, 0.0980]");
        matlabPlotColorsMap.put(ActiveLearningMethod.CONSTRAINT_PROPAGATION_HEURISTIC, "[0.9290, 0.6940, 0.1250]");
    }

    private final int numberOfExamplesToPickPerIteration;
    private final int maximumNumberOfIterations;
    private final ActiveLearningMethod activeLearningMethod;
    private final ExamplePickingMethod examplePickingMethod;
    private final Set<Label> labels;
    private final Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet;
    private final Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> predictedDataSet;
    private final Set<ResultType> resultTypes;
    private final ConstraintSet constraints;

    private ConstrainedLearningWithoutReTraining(int numberOfExamplesToPickPerIteration,
                                                 int maximumNumberOfIterations,
                                                 ActiveLearningMethod activeLearningMethod,
                                                 ExamplePickingMethod examplePickingMethod,
                                                 String workingDirectory,
                                                 ImportedDataSet importedDataSet,
                                                 Set<ResultType> resultTypes) {
        this.numberOfExamplesToPickPerIteration = numberOfExamplesToPickPerIteration;
        this.maximumNumberOfIterations = maximumNumberOfIterations;
        this.activeLearningMethod = activeLearningMethod;
        this.examplePickingMethod = examplePickingMethod;
        this.labels = importedDataSet.labels;
        this.dataSet = importedDataSet.evaluationDataSet;                   // TODO: Temporary fix.
        this.predictedDataSet = importedDataSet.evaluationPredictedDataSet; // TODO: Temporary fix.
        this.resultTypes = resultTypes;
        logger.info("Importing constraints...");
        constraints = importConstraints(workingDirectory);
    }

    private ExperimentResults runExperiment() {
        logger.info("Running experiment...");
        Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> classifierDataSet = new HashMap<>();
        Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> testingDataSet = new HashMap<>();
        Map<DataInstance<Vector>, Map<Label, Double>> selectionDataSet = new HashMap<>();
        for (Label label : labels) {
            DataSet<PredictedDataInstance<Vector, Double>> classifierDataSetCopy = new DataSetInMemory<>();
            DataSet<PredictedDataInstance<Vector, Double>> testingDataSetCopy = new DataSetInMemory<>();
            for (PredictedDataInstance<Vector, Double> instance : predictedDataSet.get(label)) {
                classifierDataSetCopy.add(new PredictedDataInstance<>(instance.name(),
                                                                      instance.features(),
                                                                      instance.label(),
                                                                      instance.source(),
                                                                      instance.probability()));
                testingDataSetCopy.add(new PredictedDataInstance<>(instance.name(),
                                                                   instance.features(),
                                                                   instance.label(),
                                                                   instance.source(),
                                                                   instance.probability()));
            }
            classifierDataSet.put(label, classifierDataSetCopy);
            testingDataSet.put(label, testingDataSetCopy);
        }
        for (Map.Entry<Label, DataSet<PredictedDataInstance<Vector, Double>>> instanceEntry : classifierDataSet.entrySet())
            for (PredictedDataInstance<Vector, Double> predictedInstance : instanceEntry.getValue()) {
                DataInstance<Vector> instance = new DataInstance<>(predictedInstance.name(),
                                                                   predictedInstance.features());
                if (!selectionDataSet.containsKey(instance))
                    selectionDataSet.put(instance, new HashMap<>());
                selectionDataSet.get(instance).put(
                        instanceEntry.getKey(),
                        predictedInstance.probability()
                );
            }
        final long experimentStartTime = System.currentTimeMillis();
        ExperimentResults results = new ExperimentResults();
        Map<DataInstance<Vector>, Map<Label, Boolean>> activeLearningDataSet = new HashMap<>();
        for (DataInstance<Vector> instance : dataSet.keySet())
            activeLearningDataSet.put(instance, new HashMap<>());
        ConstrainedLearning.Builder learningBuilder =
                new ConstrainedLearning.Builder(activeLearningDataSet)
                        .activeLearningMethod(activeLearningMethod)
                        .addConstraints(constraints.getConstraints());
        learningBuilder.addLabels(labels);
        Learning learning = learningBuilder.build();
        int numberOfExamplesPicked = 0;
        int previousNumberOfExamplesPicked = 0;
        int iterationNumber = 0;
        while (true) {
            PrecisionRecall<Vector, Double> fullPrecisionRecall = new PrecisionRecall<>(1000);
            PrecisionRecall<Vector, Double> testingPrecisionRecall = new PrecisionRecall<>(1000);
            labels.stream().forEach(label -> {
                if (resultTypes.contains(ResultType.AVERAGE_AUC_FULL_DATA_SET))
                    fullPrecisionRecall.addResult(label.getName(),
                                                  classifierDataSet.get(label),
                                                  dataInstance -> dataSet.get(new DataInstance<>(dataInstance.name(),
                                                                                                 dataInstance.features())).get(label));
                if (resultTypes.contains(ResultType.AVERAGE_AUC_TESTING_DATA_SET))
                    testingPrecisionRecall.addResult(label.getName(),
                                                     testingDataSet.get(label),
                                                     dataInstance -> dataSet.get(new DataInstance<>(dataInstance.name(),
                                                                                                    dataInstance.features())).get(label));

//                StringJoiner predictedStringJoiner = new StringJoiner(",", "[", "]");
//                StringJoiner targetStringJoiner = new StringJoiner(",", "[", "]");
//                for (PredictedDataInstance<Vector, Double> dataInstance : classifierDataSet.get(label)) {
//                    predictedStringJoiner.add("" + dataInstance.probability());
//                    targetStringJoiner.add(dataSet.get(new DataInstance<>(dataInstance.name(),
//                                                                          dataInstance.features())).get(label) ? "1" : "0");
//                }
//                logger.info("PR " + label.getName() + " predicted:\t" + predictedStringJoiner.toString());
//                logger.info("PR " + label.getName() + " target:\t" + targetStringJoiner.toString());
            });
            if (resultTypes.contains(ResultType.AVERAGE_AUC_FULL_DATA_SET))
                results.averageAreasUnderTheCurve.put(iterationNumber,
                                                      fullPrecisionRecall.getAreaUnderCurves()
                                                              .stream()
                                                              .mapToDouble(area -> area == null ? 0.0 : area.isNaN() ? 0.0 : area)
                                                              .average()
                                                              .orElse(0));
            if (resultTypes.contains(ResultType.AVERAGE_AUC_TESTING_DATA_SET))
                results.averageTestingAreasUnderTheCurve.put(iterationNumber,
                                                             testingPrecisionRecall.getAreaUnderCurves()
                                                                     .stream()
                                                                     .mapToDouble(area -> area == null ? 0.0 : area.isNaN() ? 0.0 : area)
                                                                     .average()
                                                                     .orElse(0));
            if (resultTypes.contains(ResultType.NUMBER_OF_EXAMPLES_PICKED))
                results.numberOfExamplesPicked.put(iterationNumber, numberOfExamplesPicked);
            final long startTime = System.nanoTime();
            switch (examplePickingMethod) {
                case BATCH:
                    List<Learning.InstanceToLabel> selectedInstances =
                            learning.pickInstancesToLabel(selectionDataSet, numberOfExamplesToPickPerIteration);
                    for (Learning.InstanceToLabel instance : selectedInstances) {
                        boolean trueLabel = dataSet.get(instance.getInstance()).get(instance.getLabel());
                        Map<Label, Boolean> fixedLabels = new HashMap<>(learning.getLabels(instance.getInstance()));
                        learning.labelInstance(instance, trueLabel);
                        Map<Label, Boolean> newFixedLabels = new HashMap<>(learning.getLabels(instance.getInstance()));
                        fixedLabels.keySet().forEach(newFixedLabels::remove);
                        for (Map.Entry<Label, Boolean> instanceLabelEntry : fixedLabels.entrySet()) {
                            for (PredictedDataInstance<Vector, Double> predictedInstance : classifierDataSet.get(instanceLabelEntry.getKey()))
                                if (predictedInstance.name().equals(instance.getInstance().name())) {
                                    testingDataSet.get(instanceLabelEntry.getKey()).remove(predictedInstance);
                                    predictedInstance.label(1.0);
                                    predictedInstance.probability(instanceLabelEntry.getValue() ? 1.0 : 0.0);
                                    selectionDataSet.get(
                                            new DataInstance<>(predictedInstance.name(),
                                                               predictedInstance.features())
                                    ).remove(instanceLabelEntry.getKey());
                                    break;
                                }
                        }
                    }
                    numberOfExamplesPicked += selectedInstances.size();
                    break;
                case PSEUDO_SEQUENTIAL:
                    for (int exampleNumber = 0; exampleNumber < numberOfExamplesToPickPerIteration; exampleNumber++) {
                        Learning.InstanceToLabel instance = learning.pickInstanceToLabel(selectionDataSet);
                        if (instance == null)
                            break;
                        boolean trueLabel = dataSet.get(instance.getInstance()).get(instance.getLabel());
                        Map<Label, Boolean> fixedLabels = new HashMap<>(learning.getLabels(instance.getInstance()));
                        learning.labelInstance(instance, trueLabel);
                        Map<Label, Boolean> newFixedLabels = new HashMap<>(learning.getLabels(instance.getInstance()));
                        fixedLabels.keySet().forEach(newFixedLabels::remove);
                        for (PredictedDataInstance<Vector, Double> predictedInstance : classifierDataSet.get(instance.getLabel()))
                            if (predictedInstance.name().equals(instance.getInstance().name())) {
                                for (Map.Entry<Label, Boolean> instanceLabelEntry : fixedLabels.entrySet()) {
                                    testingDataSet.get(instanceLabelEntry.getKey()).remove(predictedInstance);
                                    predictedInstance.label(1.0);
                                    predictedInstance.probability(instanceLabelEntry.getValue() ? 1.0 : 0.0);
                                }
                                break;
                            }
                        selectionDataSet.get(instance.getInstance()).remove(instance.getLabel());
                        numberOfExamplesPicked++;
                    }
                    break;
            }
            final long endTime = System.nanoTime();
            if (resultTypes.contains(ResultType.ACTIVE_LEARNING_METHOD_TIMES))
                results.activeLearningMethodTimesTaken.put(iterationNumber, endTime - startTime);
            if (numberOfExamplesPicked == previousNumberOfExamplesPicked // i.e., no examples picked in this iteration implies that there are no more unlabeled examples.
                    || ++iterationNumber >= maximumNumberOfIterations)
                break;
            previousNumberOfExamplesPicked = numberOfExamplesPicked;
        }
        final long experimentEndTime = System.currentTimeMillis();
        results.timeTaken = experimentEndTime - experimentStartTime;
        return results;
    }

    private ConstraintSet importConstraints(String workingDirectory) {
        Set<Constraint> constraints = new HashSet<>();
        try {
            Files.newBufferedReader(Paths.get(workingDirectory + "/constraints.txt")).lines().forEach(line -> {
                if (line.startsWith("!")) {
                    constraints.add(new MutualExclusionConstraint(Arrays.asList(line.substring(1).split(","))
                                                                          .stream()
                                                                          .map(Label::new)
                                                                          .collect(Collectors.toSet())));
                } else {
                    String[] lineParts = line.split(" -> ");
                    constraints.add(new SubsumptionConstraint(new Label(lineParts[0]), new Label(lineParts[1])));
                }
            });
        } catch (IOException e) {
            throw new IllegalArgumentException("There was a problem with the provided labeled noun phrases file.");
        }
        return new ConstraintSet(constraints);
    }

    private static void exportResults(int numberOfExperimentRepetitions,
                                      int numberOfExamplesToPickPerIteration,
                                      int maximumNumberOfIterations,
                                      Map<ActiveLearningMethod, List<ExperimentResults>> results,
                                      String filePath,
                                      Set<ResultType> resultTypes,
                                      boolean includeTitle,
                                      boolean includeHorizontalAxisLabel,
                                      boolean includeVerticalAxisLabel,
                                      boolean includeLegend) {
        try {
            FileWriter helperFileWriter = new FileWriter(filePath.substring(0, filePath.lastIndexOf("/")) + "/shadedErrorBar.m");
            helperFileWriter.write(shadedErrorBarMatlabCode);
            helperFileWriter.close();
            FileWriter writer = new FileWriter(filePath);
            writer.write("% numberOfExperimentRepetitions = " + numberOfExperimentRepetitions + "\n"
                                 + "% numberOfExamplesToPickPerIteration = " + numberOfExamplesToPickPerIteration + "\n"
                                 + "% maximumNumberOfIterations = " + maximumNumberOfIterations + "\n");
            int largestVectorSize = 0;
            for (ExperimentResults result : results.values().stream().flatMap(Collection::stream).collect(Collectors.toList()))
                largestVectorSize = Math.max(largestVectorSize, result.averageAreasUnderTheCurve.keySet().size());
            StringJoiner xStringJoiner = new StringJoiner(",", "[", "]");
            for (int xIndex = 0; xIndex < largestVectorSize; xIndex++)
                xStringJoiner.add(String.valueOf(xIndex));
            writer.write("x = " + xStringJoiner.toString() + ";\n");
            for (Map.Entry<ActiveLearningMethod, List<ExperimentResults>> resultsEntry : results.entrySet()) {
                String methodName = resultsEntry.getKey().name().toLowerCase();
                writer.write("% " + resultsEntry.getKey().name() + "\n");
                if (resultTypes.contains(ResultType.AVERAGE_AUC_FULL_DATA_SET)) {
                    writer.write("x_" + methodName + "_" + ResultType.AVERAGE_AUC_FULL_DATA_SET.name().toLowerCase() + " = zeros(" + resultsEntry.getValue().size() + ", " + largestVectorSize + ");\n");
                    writer.write("y_" + methodName + "_" + ResultType.AVERAGE_AUC_FULL_DATA_SET.name().toLowerCase() + " = zeros(" + resultsEntry.getValue().size() + ", " + largestVectorSize + ");\n");
                }
                if (resultTypes.contains(ResultType.AVERAGE_AUC_TESTING_DATA_SET)) {
                    writer.write("x_" + methodName + "_" + ResultType.AVERAGE_AUC_TESTING_DATA_SET.name().toLowerCase() + " = zeros(" + resultsEntry.getValue().size() + ", " + largestVectorSize + ");\n");
                    writer.write("y_" + methodName + "_" + ResultType.AVERAGE_AUC_TESTING_DATA_SET.name().toLowerCase() + " = zeros(" + resultsEntry.getValue().size() + ", " + largestVectorSize + ");\n");
                }
                if (resultTypes.contains(ResultType.AVERAGE_AUC_EVALUATION_DATA_SET)) {
                    writer.write("x_" + methodName + "_" + ResultType.AVERAGE_AUC_EVALUATION_DATA_SET.name().toLowerCase() + " = zeros(" + resultsEntry.getValue().size() + ", " + largestVectorSize + ");\n");
                    writer.write("y_" + methodName + "_" + ResultType.AVERAGE_AUC_EVALUATION_DATA_SET.name().toLowerCase() + " = zeros(" + resultsEntry.getValue().size() + ", " + largestVectorSize + ");\n");
                }
                if (resultTypes.contains(ResultType.NUMBER_OF_EXAMPLES_PICKED)) {
                    writer.write("x_" + methodName + "_" + ResultType.NUMBER_OF_EXAMPLES_PICKED.name().toLowerCase() + " = zeros(" + resultsEntry.getValue().size() + ", " + largestVectorSize + ");\n");
                    writer.write("y_" + methodName + "_" + ResultType.NUMBER_OF_EXAMPLES_PICKED.name().toLowerCase() + " = zeros(" + resultsEntry.getValue().size() + ", " + largestVectorSize + ");\n");
                }
                if (resultTypes.contains(ResultType.ACTIVE_LEARNING_METHOD_TIMES)) {
                    writer.write("x_" + methodName + "_" + ResultType.ACTIVE_LEARNING_METHOD_TIMES.name().toLowerCase() + " = zeros(" + resultsEntry.getValue().size() + ", " + largestVectorSize + ");\n");
                    writer.write("y_" + methodName + "_" + ResultType.ACTIVE_LEARNING_METHOD_TIMES.name().toLowerCase() + " = zeros(" + resultsEntry.getValue().size() + ", " + largestVectorSize + ");\n");
                }
                if (resultTypes.contains(ResultType.TOTAL_TIME_TAKEN))
                    writer.write(ResultType.TOTAL_TIME_TAKEN.name().toLowerCase() + "_" + methodName + " = zeros(" + resultsEntry.getValue().size() + ", 1);\n");
                int experimentIndex = 1;
                for (ExperimentResults result : resultsEntry.getValue()) {
                    writer.write("% Experiment " + experimentIndex + ":\n");
                    if (resultTypes.contains(ResultType.AVERAGE_AUC_FULL_DATA_SET))
                        writer.write(simpleMapToMatlabString(result.averageAreasUnderTheCurve, methodName, ResultType.AVERAGE_AUC_FULL_DATA_SET.name().toLowerCase(), "1.0", experimentIndex, largestVectorSize) + "\n");
                    if (resultTypes.contains(ResultType.AVERAGE_AUC_TESTING_DATA_SET))
                        writer.write(simpleMapToMatlabString(result.averageTestingAreasUnderTheCurve, methodName, ResultType.AVERAGE_AUC_TESTING_DATA_SET.name().toLowerCase(), "1.0", experimentIndex, largestVectorSize) + "\n");
                    if (resultTypes.contains(ResultType.AVERAGE_AUC_EVALUATION_DATA_SET))
                        writer.write(simpleMapToMatlabString(result.averageEvaluationAreasUnderTheCurve, methodName, ResultType.AVERAGE_AUC_EVALUATION_DATA_SET.name().toLowerCase(), null, experimentIndex, largestVectorSize) + "\n");
                    if (resultTypes.contains(ResultType.NUMBER_OF_EXAMPLES_PICKED))
                        writer.write(simpleMapToMatlabString(result.numberOfExamplesPicked, methodName, ResultType.NUMBER_OF_EXAMPLES_PICKED.name().toLowerCase(), null, experimentIndex, largestVectorSize) + "\n");
                    if (resultTypes.contains(ResultType.ACTIVE_LEARNING_METHOD_TIMES))
                        writer.write(simpleMapToMatlabString(result.activeLearningMethodTimesTaken, methodName, ResultType.ACTIVE_LEARNING_METHOD_TIMES.name().toLowerCase(), "0.0", experimentIndex, largestVectorSize) + "\n");
                    if (resultTypes.contains(ResultType.TOTAL_TIME_TAKEN))
                        writer.write("times_" + methodName + "(" + experimentIndex + ") = " + (int) Math.floor(result.timeTaken / 1000) + ";\n");
                    experimentIndex++;
                }
            }
            writer.write("\n% Plot results\n");
            writer.write("figure;\n");
            int plotIndex = 1;
            boolean totalTimeTakenResultTypeRemoved = resultTypes.contains(ResultType.TOTAL_TIME_TAKEN);
            resultTypes.remove(ResultType.TOTAL_TIME_TAKEN);
            for (ResultType resultTypePlot : resultTypes) {
                writer.write("subplot(1, " + resultTypes.size() + ", " + plotIndex + ");\n");
                writer.write("hold on;\n");
                for (Map.Entry<ActiveLearningMethod, List<ExperimentResults>> resultsEntry : results.entrySet()) {
                    String methodName = resultsEntry.getKey().name().toLowerCase();
                    if (plotIndex == 1)
                        writer.write("H(" + (resultsEntry.getKey().ordinal() + 1) + ") = shadedErrorBar(x, " +
                                             "y_" + methodName + "_" + resultTypePlot.name().toLowerCase() + ", " +
                                             "{@(x) mean(x, 1), @(x) [max(x, [], 1) - mean(x, 1); mean(x, 1) - min(x,[], 1)]}, " +
                                             "{'Color', " + matlabPlotColorsMap.get(resultsEntry.getKey()) + ", 'LineWidth', 3}, 1);\n");
                    else
                        writer.write("shadedErrorBar(x, " +
                                             "y_" + methodName + "_" + resultTypePlot.name().toLowerCase() + ", " +
                                             "{@(x) mean(x, 1), @(x) [max(x, [], 1) - mean(x, 1); mean(x, 1) - min(x,[], 1)]}, " +
                                             "{'Color', " + matlabPlotColorsMap.get(resultsEntry.getKey()) + ", 'LineWidth', 3}, 1);\n");
                }
                if (includeTitle)
                    switch (resultTypePlot) {
                        case AVERAGE_AUC_FULL_DATA_SET:
                            writer.write("title('Average AUC Over Full Data Set');\n");
                            break;
                        case AVERAGE_AUC_TESTING_DATA_SET:
                            writer.write("title('Average AUC Over Unlabeled Data Set');\n");
                            break;
                        case AVERAGE_AUC_EVALUATION_DATA_SET:
                            writer.write("title('Average AUC Over Evaluation Data Set');\n");
                            break;
                        case NUMBER_OF_EXAMPLES_PICKED:
                            writer.write("title('Number of Labels Requested Per Iteration');\n");
                            break;
                        case ACTIVE_LEARNING_METHOD_TIMES:
                            writer.write("title('Time Spent in Active Learning Method Per Iteration');\n");
                            break;
                    }
                writer.write("ylim([0 1]);\n");
                if (includeHorizontalAxisLabel)
                    writer.write("xlabel('Iteration Number', 'FontSize', 22);\n");
                if (includeVerticalAxisLabel)
                    switch (resultTypePlot) {
                        case AVERAGE_AUC_FULL_DATA_SET:
                        case AVERAGE_AUC_TESTING_DATA_SET:
                        case AVERAGE_AUC_EVALUATION_DATA_SET:
                            writer.write("ylabel('Average AUC', 'FontSize', 22);\n");
                            break;
                        case NUMBER_OF_EXAMPLES_PICKED:
                            writer.write("ylabel('Number of Labels Requested', 'FontSize', 22);\n");
                            break;
                        case ACTIVE_LEARNING_METHOD_TIMES:
                            writer.write("ylabel('Time Spent in Active Learning Method', 'FontSize', 22);\n");
                            break;
                    }
                writer.write("set(gca, 'FontSize', 22);\n");
                writer.write("hold off;\n");
                plotIndex++;
            }
            if (totalTimeTakenResultTypeRemoved)
                resultTypes.add(ResultType.TOTAL_TIME_TAKEN);
            if (includeLegend) {
                StringJoiner legendPlotNames = new StringJoiner(", ", "[", "]");
                StringJoiner legendPlotDescriptions = new StringJoiner(", ", "{", "}");
                for (ActiveLearningMethod method : results.keySet()) {
                    legendPlotNames.add("H(" + (method.ordinal() + 1) + ").mainLine");
                    legendPlotDescriptions.add("'" + toTitleCase(method.name().toLowerCase().replace("_", " ")) + "'");
                }
                writer.write("legend(" + legendPlotNames.toString() + ", " + legendPlotDescriptions.toString() + ", "
                                     + "'Location', 'Southeast');\n");
            }
            writer.write("set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [-0.25 -0.1 6.5 5], 'PaperSize', [6 4.8]);\n" +
                                 "saveas(gcf, 'results.pdf');");
            writer.close();
        } catch (IOException e) {
            logger.error("An exception was thrown while trying to export a set of experiment results.", e);
        }
    }

    private static String toTitleCase(String input) {
        StringBuilder titleCase = new StringBuilder();
        boolean nextTitleCase = true;
        for (char c : input.toCharArray()) {
            if (Character.isSpaceChar(c)) {
                nextTitleCase = true;
            } else if (nextTitleCase) {
                c = Character.toTitleCase(c);
                nextTitleCase = false;
            }
            titleCase.append(c);
        }
        return titleCase.toString();
    }

    private static String simpleMapToMatlabString(Map<Integer, ?> map,
                                                  String methodName,
                                                  String variableNameSuffix,
                                                  String fillingValue,
                                                  int experimentIndex,
                                                  int largestVectorSize) {
        StringJoiner indexesStringJoiner = new StringJoiner(", ", "[", "]");
        StringJoiner valuesStringJoiner = new StringJoiner(", ", "[", "]");
        int[] largestIndex = new int[] { 0 };
        Object[] lastValue = new Object[] { null };
        map.entrySet()
                .stream()
                .sorted(Map.Entry.comparingByKey(Integer::compareTo))
                .forEachOrdered(entry -> {
                    if (entry.getKey() > largestIndex[0]) {
                        largestIndex[0] = entry.getKey();
                        lastValue[0] = entry.getValue();
                    }
                    indexesStringJoiner.add(String.valueOf(entry.getKey()));
                    valuesStringJoiner.add(String.valueOf(entry.getValue()));
                });
        largestIndex[0]++;
        for (; largestIndex[0] < largestVectorSize; largestIndex[0]++) {
            indexesStringJoiner.add(String.valueOf(largestIndex[0]));
            if (fillingValue != null)
                valuesStringJoiner.add(fillingValue);
            else
                valuesStringJoiner.add(String.valueOf(lastValue[0]));
        }
        return variableName("x", methodName, variableNameSuffix) + "(" + experimentIndex + ", :) = " + indexesStringJoiner.toString() + ";\n" +
                variableName("y", methodName, variableNameSuffix) + "(" + experimentIndex + ", :) = " + valuesStringJoiner.toString() + ";";
    }

    private static String variableName(String prefix, String methodName, String suffix) {
        return prefix + "_" + methodName + "_" + suffix;
    }

    private static class ImportedDataSet {
        private final Set<Label> labels;
        private final Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet;
        private final Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> predictedDataSet;
        private final Map<DataInstance<Vector>, Map<Label, Boolean>> evaluationDataSet;
        private final Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> evaluationPredictedDataSet;
        private final Map<Label, DataSetStatistics> statistics;
        private final Map<Label, TrainableClassifier<Vector, Double>> classifiers;

        public ImportedDataSet(Set<Label> labels,
                               Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet,
                               Map<DataInstance<Vector>, Map<Label, Boolean>> evaluationDataSet) {
            this(labels, dataSet, evaluationDataSet, 0.0, 0.0);
        }

        public ImportedDataSet(Set<Label> labels,
                               Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet,
                               Map<DataInstance<Vector>, Map<Label, Boolean>> evaluationDataSet,
                               double l1RegularizationWeight,
                               double l2RegularizationWeight) {
            this.labels = labels;
            this.dataSet = dataSet;
            this.evaluationDataSet = evaluationDataSet;
            statistics = new HashMap<>();
            for (Label label : labels)
                statistics.put(label, new DataSetStatistics());
            for (Map.Entry<DataInstance<Vector>, Map<Label, Boolean>> instanceEntry : dataSet.entrySet()) {
                for (Map.Entry<Label, Boolean> instanceLabelEntry : instanceEntry.getValue().entrySet()) {
                    if (instanceLabelEntry.getValue())
                        statistics.get(instanceLabelEntry.getKey()).numberOfPositiveExamples++;
                    else
                        statistics.get(instanceLabelEntry.getKey()).numberOfNegativeExamples++;
                    statistics.get(instanceLabelEntry.getKey()).totalNumberOfExamples++;
                }
            }
            predictedDataSet = new HashMap<>();
            evaluationPredictedDataSet = new HashMap<>();
            classifiers = new HashMap<>();
            for (Label label : labels) {
                predictedDataSet.put(label, new DataSetInMemory<>());
                evaluationPredictedDataSet.put(label, new DataSetInMemory<>());
                Vector randomFeatureVector = dataSet.keySet().iterator().next().features();
                LogisticRegressionAdaGrad.Builder classifierBuilder =
                        new LogisticRegressionAdaGrad.Builder(randomFeatureVector.size())
                                .useBiasTerm(true)
                                .l1RegularizationWeight(l1RegularizationWeight)
                                .l2RegularizationWeight(l2RegularizationWeight)
                                .loggingLevel(0)
                                .sampleWithReplacement(true)
                                .maximumNumberOfIterations(1000)
                                .maximumNumberOfIterationsWithNoPointChange(10)
                                .pointChangeTolerance(1e-5)
                                .checkForPointConvergence(true)
                                .batchSize(100);
                if (randomFeatureVector instanceof DenseVector)
                    classifierBuilder.sparse(false);
                else
                    classifierBuilder.sparse(true);
                classifiers.put(label, classifierBuilder.build());
            }
            for (Map.Entry<DataInstance<Vector>, Map<Label, Boolean>> instanceEntry : dataSet.entrySet())
                for (Map.Entry<Label, Boolean> instanceLabelEntry : instanceEntry.getValue().entrySet()) {
                    PredictedDataInstance<Vector, Double> predictedInstance =
                            new PredictedDataInstance<>(instanceEntry.getKey().name(),
                                                        instanceEntry.getKey().features(),
                                                        instanceLabelEntry.getValue() ? 1.0 : 0.0,
                                                        null,
                                                        1.0);
                    predictedDataSet.get(instanceLabelEntry.getKey()).add(predictedInstance);
                }
            for (Map.Entry<DataInstance<Vector>, Map<Label, Boolean>> instanceEntry : evaluationDataSet.entrySet())
                for (Map.Entry<Label, Boolean> instanceLabelEntry : instanceEntry.getValue().entrySet()) {
                    PredictedDataInstance<Vector, Double> predictedInstance =
                            new PredictedDataInstance<>(instanceEntry.getKey().name(),
                                                        instanceEntry.getKey().features(),
                                                        instanceLabelEntry.getValue() ? 1.0 : 0.0,
                                                        null,
                                                        1.0);
                    evaluationPredictedDataSet.get(instanceLabelEntry.getKey()).add(predictedInstance);
                }
            labels.stream().forEach(label -> {
                classifiers.get(label).train(predictedDataSet.get(label));
                classifiers.get(label).predictInPlace(evaluationPredictedDataSet.get(label));
                for (PredictedDataInstance<Vector, Double> instance : evaluationPredictedDataSet.get(label))
                    if (instance.label() < 0.5) {
                        instance.label(1 - instance.label());
                        instance.probability(1 - instance.probability());
                    }
            });
        }

        public Set<Label> getLabels() {
            return labels;
        }

        public Map<DataInstance<Vector>, Map<Label, Boolean>> getDataSet() {
            return dataSet;
        }

        public Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> getEvaluationPredictedDataSet() {
            return evaluationPredictedDataSet;
        }

        public Map<Label, DataSetStatistics> getStatistics() {
            return statistics;
        }

        public Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> getPredictedDataSet() {
            return predictedDataSet;
        }

        public Map<Label, TrainableClassifier<Vector, Double>> getClassifiers() {
            return classifiers;
        }

        public void exportStatistics(String filePath) {
            try {
                FileWriter writer = new FileWriter(filePath);
                for (Label label : labels)
                    writer.write(label.getName() + ": {" + statistics.get(label).toString() + " }\n");
                writer.close();
            } catch (IOException e) {
                logger.error("An exception was thrown while trying to export a set of data set statistics.", e);
            }
        }
    }

    private static class DataSetStatistics {
        private int numberOfPositiveExamples = 0;
        private int numberOfNegativeExamples = 0;
        private int totalNumberOfExamples = 0;

        @Override
        public String toString() {
            return "Positive: " + numberOfPositiveExamples
                    + "\tNegative: " + numberOfNegativeExamples
                    + "\tTotal: " + totalNumberOfExamples;
        }
    }

    private enum ExamplePickingMethod {
        PSEUDO_SEQUENTIAL,
        BATCH
    }

    private enum ResultType {
        AVERAGE_AUC_FULL_DATA_SET,
        AVERAGE_AUC_TESTING_DATA_SET,
        AVERAGE_AUC_EVALUATION_DATA_SET,
        NUMBER_OF_EXAMPLES_PICKED,
        ACTIVE_LEARNING_METHOD_TIMES,
        TOTAL_TIME_TAKEN
    }

    private static class ExperimentResults {
        private Map<Integer, Double> averageAreasUnderTheCurve = new HashMap<>();
        private Map<Integer, Double> averageTestingAreasUnderTheCurve = new HashMap<>();
        private Map<Integer, Double> averageEvaluationAreasUnderTheCurve = new HashMap<>();
        private Map<Integer, Integer> numberOfExamplesPicked = new HashMap<>();
        private Map<Integer, Long> activeLearningMethodTimesTaken = new HashMap<>();
        private long timeTaken;
    }

    private static ImportedDataSet importISOLETDataSet(String workingDirectory,
                                                       double l1RegularizationWeight,
                                                       double l2RegularizationWeight) {
        logger.info("Importing ISOLET data set...");
        Map<Vector, String> labeledInstances = new HashMap<>();
        try {
            Files.newBufferedReader(Paths.get(workingDirectory + "/labeled_data.csv")).lines().forEach(line -> {
                String[] lineParts = line.split(",");
                if (lineParts.length > 1) {
                    double[] featureValues = new double[lineParts.length - 1];
                    for (int linePartIndex = 0; linePartIndex < lineParts.length - 1; linePartIndex++)
                        featureValues[linePartIndex] = Double.parseDouble(lineParts[linePartIndex]);
                    labeledInstances.put(Vectors.dense(featureValues), lineParts[lineParts.length - 1]);
                }
            });
        } catch (IOException e) {
            throw new IllegalArgumentException("There was a problem with the provided labeled noun phrases file.");
        }
        Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet = new HashMap<>();
        Set<Label> labels = labeledInstances.values().stream().map(Label::new).collect(Collectors.toSet());
        Set<String> uniqueNames = new HashSet<>();
        for (Map.Entry<Vector, String> labeledInstanceEntry : labeledInstances.entrySet()) {
            Vector features = labeledInstanceEntry.getKey();
            String name = labeledInstanceEntry.getValue() + ":" + features.toString();
            DataInstance<Vector> dataInstance = new DataInstance<>(name, features);
            if (!uniqueNames.contains(name)) {
                uniqueNames.add(name);
                String labelName = labeledInstanceEntry.getValue();
                Set<String> negativeLabels = labels.stream().map(Label::getName).collect(Collectors.toSet());
                negativeLabels.remove(labelName);
                if (!dataSet.containsKey(dataInstance))
                    dataSet.put(dataInstance, new HashMap<>());
                dataSet.get(dataInstance).put(new Label(labelName), true);
                for (String negativeLabelName : negativeLabels)
                    dataSet.get(dataInstance).put(new Label(negativeLabelName), false);
            }
        }
        Map<DataInstance<Vector>, Map<Label, Boolean>> evaluationDataSet = new HashMap<>();
        try {
            Set<String> evaluationUniqueNames = new HashSet<>();
            Files.newBufferedReader(Paths.get(workingDirectory + "/evaluation_data.csv")).lines().forEach(line -> {
                String[] lineParts = line.split(",");
                if (lineParts.length > 1) {
                    double[] featureValues = new double[lineParts.length - 1];
                    for (int linePartIndex = 0; linePartIndex < lineParts.length - 1; linePartIndex++)
                        featureValues[linePartIndex] = Double.parseDouble(lineParts[linePartIndex]);
                    Vector features = Vectors.dense(featureValues);
                    String name = features.toString();
                    DataInstance<Vector> dataInstance = new DataInstance<>(name, features);
                    if (!evaluationUniqueNames.contains(name)) {
                        evaluationUniqueNames.add(name);
                        String labelName = lineParts[lineParts.length - 1];
                        Set<String> negativeLabels = labels.stream().map(Label::getName).collect(Collectors.toSet());
                        negativeLabels.remove(labelName);
                        if (!evaluationDataSet.containsKey(dataInstance))
                            evaluationDataSet.put(dataInstance, new HashMap<>());
                        evaluationDataSet.get(dataInstance).put(new Label(labelName), true);
                        for (String negativeLabelName : negativeLabels)
                            evaluationDataSet.get(dataInstance).put(new Label(negativeLabelName), false);
                    }
                }
            });
        } catch (IOException e) {
            throw new IllegalArgumentException("There was a problem with the provided labeled noun phrases file.");
        }
        return new ImportedDataSet(labels, dataSet, evaluationDataSet, l1RegularizationWeight, l2RegularizationWeight);
    }

    private static ImportedDataSet importLIBSVMDataSet(String workingDirectory,
                                                       boolean sparseFeatures,
                                                       double l1RegularizationWeight,
                                                       double l2RegularizationWeight) {
        logger.info("Importing LIBSVM data set...");
        Map<Vector, String> labeledInstances = new HashMap<>();
        final int[] largestVectorIndex = { 0 };
        try {
            Files.newBufferedReader(Paths.get(workingDirectory + "/labeled_data.csv")).lines().forEach(line -> {
                String[] lineParts = line.split(",");
                if (lineParts.length > 1) {
                    for (int linePartIndex = 1; linePartIndex < lineParts.length; linePartIndex++)
                        largestVectorIndex[0] = Math.max(largestVectorIndex[0],
                                                         Integer.parseInt(lineParts[linePartIndex].split(":")[0]));
                }
            });
            Files.newBufferedReader(Paths.get(workingDirectory + "/evaluation_data.csv")).lines().forEach(line -> {
                String[] lineParts = line.split(",");
                if (lineParts.length > 1) {
                    for (int linePartIndex = 1; linePartIndex < lineParts.length; linePartIndex++)
                        largestVectorIndex[0] = Math.max(largestVectorIndex[0],
                                                         Integer.parseInt(lineParts[linePartIndex].split(":")[0]));
                }
            });
            Files.newBufferedReader(Paths.get(workingDirectory + "/labeled_data.csv")).lines().forEach(line -> {
                String[] lineParts = line.split(",");
                if (lineParts.length > 1) {
                    Vector features;
                    if (sparseFeatures)
                        features = Vectors.sparse(largestVectorIndex[0]);
                    else
                        features = Vectors.dense(largestVectorIndex[0]);
                    for (int linePartIndex = 1; linePartIndex < lineParts.length; linePartIndex++)
                        features.set(Integer.parseInt(lineParts[linePartIndex].split(":")[0]) - 1,
                                     Double.parseDouble(lineParts[linePartIndex].split(":")[1]));
                    labeledInstances.put(features, lineParts[0]);
                }
            });
        } catch (IOException e) {
            throw new IllegalArgumentException("There was a problem with the provided labeled noun phrases file.");
        }
        Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet = new HashMap<>();
        Set<Label> labels = labeledInstances.values().stream().map(Label::new).collect(Collectors.toSet());
        Set<String> uniqueNames = new HashSet<>();
        for (Map.Entry<Vector, String> labeledInstanceEntry : labeledInstances.entrySet()) {
            Vector features = labeledInstanceEntry.getKey();
            String name = features.toString();
            DataInstance<Vector> dataInstance = new DataInstance<>(name, features);
            if (!uniqueNames.contains(name)) {
                uniqueNames.add(name);
                String labelName = labeledInstanceEntry.getValue();
                Set<String> negativeLabels = labels.stream().map(Label::getName).collect(Collectors.toSet());
                negativeLabels.remove(labelName);
                if (!dataSet.containsKey(dataInstance))
                    dataSet.put(dataInstance, new HashMap<>());
                dataSet.get(dataInstance).put(new Label(labelName), true);
                for (String negativeLabelName : negativeLabels)
                    dataSet.get(dataInstance).put(new Label(negativeLabelName), false);
            }
        }
        Map<DataInstance<Vector>, Map<Label, Boolean>> evaluationDataSet = new HashMap<>();
        try {
            Set<String> evaluationUniqueNames = new HashSet<>();
            Files.newBufferedReader(Paths.get(workingDirectory + "/evaluation_data.csv")).lines().forEach(line -> {
                String[] lineParts = line.split(",");
                if (lineParts.length > 1) {
                    Vector features;
                    if (sparseFeatures)
                        features = Vectors.sparse(largestVectorIndex[0]);
                    else
                        features = Vectors.dense(largestVectorIndex[0]);
                    for (int linePartIndex = 1; linePartIndex < lineParts.length; linePartIndex++)
                        features.set(Integer.parseInt(lineParts[linePartIndex].split(":")[0]) - 1,
                                     Double.parseDouble(lineParts[linePartIndex].split(":")[1]));
                    String name = features.toString();
                    DataInstance<Vector> dataInstance = new DataInstance<>(name, features);
                    if (!evaluationUniqueNames.contains(name)) {
                        evaluationUniqueNames.add(name);
                        String labelName = lineParts[0];
                        Set<String> negativeLabels = labels.stream().map(Label::getName).collect(Collectors.toSet());
                        negativeLabels.remove(labelName);
                        if (!evaluationDataSet.containsKey(dataInstance))
                            evaluationDataSet.put(dataInstance, new HashMap<>());
                        evaluationDataSet.get(dataInstance).put(new Label(labelName), true);
                        for (String negativeLabelName : negativeLabels)
                            evaluationDataSet.get(dataInstance).put(new Label(negativeLabelName), false);
                    }
                }
            });
        } catch (IOException e) {
            throw new IllegalArgumentException("There was a problem with the provided labeled noun phrases file.");
        }
        return new ImportedDataSet(labels, dataSet, evaluationDataSet, l1RegularizationWeight, l2RegularizationWeight);
    }

    private static ImportedDataSet importNELLDataSet(String cplFeatureMapDirectory,
                                                     String workingDirectory,
                                                     double l1RegularizationWeight,
                                                     double l2RegularizationWeight) { // TODO: Fix this one with respect to evaluation data.
        logger.info("Importing NELL data set...");
        Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet = new HashMap<>();
        logger.info("Importing NELL labeled noun phrases...");
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
        logger.info("Importing NELL feature map...");
        Map<String, Vector> featureMap;
        if (Files.exists(Paths.get(workingDirectory + "/features.bin")))
            featureMap = Utilities.readMap(workingDirectory + "/features.bin");
        else
            featureMap = buildFeatureMap(workingDirectory + "/features.bin",
                                         cplFeatureMapDirectory,
                                         labeledNounPhrases.keySet());
        Set<Label> labels =
                labeledNounPhrases.values()
                        .stream()
                        .flatMap(Collection::stream)
                        .map(Label::new)
                        .collect(Collectors.toSet());
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
                continue;
            } else {
                filteredLabeledNounPhrases.put(nounPhrase, positiveLabels);
                features = featureMap.get(nounPhrase);
            }
            DataInstance<Vector> dataInstance = new DataInstance<>(nounPhrase, features);
            if (!dataSet.containsKey(dataInstance))
                dataSet.put(dataInstance, new HashMap<>());
            for (String labelName : positiveLabels)
                dataSet.get(dataInstance).put(new Label(labelName), true);
            for (String labelName : negativeLabels)
                dataSet.get(dataInstance).put(new Label(labelName), false);
        }
        logger.info("NELL noun phrases without features that were ignored: " + nounPhrasesWithoutFeatures);
        exportLabeledNounPhrases(filteredLabeledNounPhrases, workingDirectory + "/filtered_labeled_nps.tsv");
        return new ImportedDataSet(labels, dataSet, null, l1RegularizationWeight, l2RegularizationWeight);
    }

    private static Map<String, Vector> buildFeatureMap(String featureMapDirectory, String cplFeatureMapDirectory) {
        return buildFeatureMap(featureMapDirectory, cplFeatureMapDirectory, null);
    }

    private static Map<String, Vector> buildFeatureMap(String featureMapDirectory,
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
            logger.error("An exception was thrown while trying to build the CPL feature map.", e);
        }
//        Utilities.writeMap(featureMap, featureMapDirectory); // TODO: Fix this.
        return featureMap;
    }

    private static Map<String, Integer> buildContextsMap(String cplFeatureMapDirectory) throws IOException {
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

    private static void exportLabeledNounPhrases(Map<String, Set<String>> labeledNounPhrases, String filePath) {
        try {
            FileWriter writer = new FileWriter(filePath);
            for (String nounPhrase : labeledNounPhrases.keySet()) {
                StringJoiner stringJoiner = new StringJoiner(",");
                labeledNounPhrases.get(nounPhrase).forEach(stringJoiner::add);
                writer.write(nounPhrase + "\t" + stringJoiner.toString() + "\n");
            }
            writer.close();
        } catch (IOException e) {
            logger.error("An exception was thrown while trying to export a set of labeled noun phrases.", e);
        }
    }

    public static Map<ActiveLearningMethod, List<ExperimentResults>> runExperiments(
            int numberOfExperimentRepetitions,
            int numberOfExamplesToPickPerIteration,
            int maximumNumberOfIterations,
            String workingDirectory,
            ActiveLearningMethod[] activeLearningMethods,
            ExamplePickingMethod examplePickingMethod,
            ImportedDataSet importedDataSet,
            Set<ResultType> resultTypes
    ) {
        Map<ActiveLearningMethod, List<ExperimentResults>> results = new HashMap<>();
        for (ActiveLearningMethod activeLearningMethod : activeLearningMethods) {
            logger.info("Running experiment for " + activeLearningMethod.name() + "...");
            results.put(activeLearningMethod, new ArrayList<>());
            ConstrainedLearningWithoutReTraining experiment = new ConstrainedLearningWithoutReTraining(
                    numberOfExamplesToPickPerIteration,
                    maximumNumberOfIterations,
                    activeLearningMethod,
                    examplePickingMethod,
                    workingDirectory,
                    importedDataSet,
                    resultTypes
            );
            Map<Integer, Double> averageAreasUnderTheCurve = new HashMap<>();
            Map<Integer, Integer> countAreasUnderTheCurve = new HashMap<>();
            int effectiveNumberOfExperimentRepetitions = activeLearningMethod.equals(ActiveLearningMethod.RANDOM) ? numberOfExperimentRepetitions : 1;
            for (int repetition = 0; repetition < effectiveNumberOfExperimentRepetitions; repetition++) {
                logger.info("Running experiment repetition " + (repetition + 1) + "...");
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
                for (int key : averageAreasUnderTheCurve.keySet())
                    averageAreasUnderTheCurve.put(key, averageAreasUnderTheCurve.get(key) / countAreasUnderTheCurve.get(key));
            }
        }
        logger.info("Finished!");
        return results;
    }

    public static void main(String[] args) {
        int numberOfExperimentRepetitions = 2;
        int numberOfExamplesToPickPerIteration = 1000000;
        int maximumNumberOfIterations = 1000000000;
        boolean includeTitleInResultsPlot = false;
        boolean includeHorizontalAxisLabelInResultsPlot = true;
        boolean includeVerticalAxisLabelInResultsPlot = true;
        boolean includeLegendInResultsPlot = false;
        ActiveLearningMethod[] activeLearningMethods = new ActiveLearningMethod[] {
                ActiveLearningMethod.RANDOM,
                ActiveLearningMethod.UNCERTAINTY_HEURISTIC,
                ActiveLearningMethod.CONSTRAINT_PROPAGATION_HEURISTIC
        };
        ExamplePickingMethod examplePickingMethod = ExamplePickingMethod.BATCH;
        Set<ResultType> resultTypes = new HashSet<>();
        resultTypes.add(ResultType.AVERAGE_AUC_FULL_DATA_SET);

        String workingDirectory;
        ImportedDataSet dataSet;
        Map<ActiveLearningMethod, List<ExperimentResults>> results;

//        // NELL Data Set Experiment
//        logger.info("Running NELL experiment...");
//        numberOfExperimentRepetitions = 1;
//        numberOfExamplesToPickPerIteration = 1;
//        maximumNumberOfIterations = 100;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment NELL";
//        String cplFeatureMapDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/Data Sets/NELL/Server/all-pairs/all-pairs-OC-2010-12-01-small200-gz";
//        dataSet = importNELLDataSet(cplFeatureMapDirectory, workingDirectory, 1.0, 1.0);

//        // IRIS Data Set Experiment
//        logger.info("Running IRIS experiment...");
//        numberOfExperimentRepetitions = 10;
//        numberOfExamplesToPickPerIteration = 1;
//        maximumNumberOfIterations = 1000;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment IRIS";
//        dataSet = importISOLETDataSet(workingDirectory, 0.1, 0.1);

//        // ABALONE Data Set Experiment
//        logger.info("Running ABALONE experiment...");
//        numberOfExperimentRepetitions = 10;
//        numberOfExamplesToPickPerIteration = 1;
//        maximumNumberOfIterations = 1000;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment ABALONE";
//        dataSet = importISOLETDataSet(workingDirectory, 0.0, 0.0);

//        // LETTER Data Set Experiment
//        logger.info("Running LETTER experiment...");
//        numberOfExperimentRepetitions = 10;
//        numberOfExamplesToPickPerIteration = 1;
//        maximumNumberOfIterations = 100000;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment LETTER-";
//        dataSet = importLIBSVMDataSet(workingDirectory, false, 0.0, 0.0);

        // PENDIGITS Data Set Experiment
        logger.info("Running PENDIGITS experiment...");
        numberOfExperimentRepetitions = 10;
        numberOfExamplesToPickPerIteration = 1;
        maximumNumberOfIterations = 100000;
        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment PENDIGITS-";
        dataSet = importLIBSVMDataSet(workingDirectory, false, 0.0, 0.0);

//        // WINE Data Set Experiment
//        logger.info("Running WINE experiment...");
//        numberOfExperimentRepetitions = 10;
//        numberOfExamplesToPickPerIteration = 1;
//        maximumNumberOfIterations = 1000;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment WINE";
//        dataSet = importLIBSVMDataSet(workingDirectory, false);

//        // SENSIT-VEHICLE-SEISMIC Data Set Experiment
//        logger.info("Running SENSIT-VEHICLE-SEISMIC experiment...");
//        numberOfExperimentRepetitions = 1;
//        initialNumberOfExamples = 1000;
//        initialRatioOfPositiveExamples = 0.3;
//        numberOfExamplesToPickPerIteration = 10000;
//        String workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment SENSIT-VEHICLE-SEISMIC";
//        ImportedDataSet sensitVehicleSeismicTypeDataSet = importLIBSVMDataSet(workingDirectory, false);
//        runExperiments(numberOfExperimentRepetitions,
//                       initialNumberOfExamples,
//                       initialRatioOfPositiveExamples,
//                       numberOfExamplesToPickPerIteration,
//                       workingDirectory,
//                       activeLearningMethods,
//                       examplePickingMethod,
//                       sensitVehicleSeismicTypeDataSet,
//                       resultTypes);
//        logger.info("Finished all experiments!");

//        // GLASS Data Set Experiment
//        logger.info("Running GLASS experiment...");
//        numberOfExperimentRepetitions = 10;
//        numberOfExamplesToPickPerIteration = 1;
//        maximumNumberOfIterations = 10000000;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment GLASS";
//        dataSet = importLIBSVMDataSet(workingDirectory, false, 0.0, 0.0);

//        // VEHICLE Data Set Experiment
//        logger.info("Running VEHICLE experiment...");
//        numberOfExperimentRepetitions = 10;
//        numberOfExamplesToPickPerIteration = 10;
//        maximumNumberOfIterations = 1000;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment VEHICLE";
//        dataSet = importLIBSVMDataSet(workingDirectory, false, 0.0, 0.0);

//        // MNIST Data Set Experiment
//        logger.info("Running MNIST experiment...");
//        numberOfExperimentRepetitions = 5;
//        numberOfExamplesToPickPerIteration = 1;
//        maximumNumberOfIterations = 500;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment MNIST";
//        dataSet = importLIBSVMDataSet(workingDirectory, true, 0.0, 0.0);

//        // VOWEL Data Set Experiment
//        logger.info("Running VOWEL experiment...");
//        numberOfExperimentRepetitions = 10;
//        numberOfExamplesToPickPerIteration = 1;
//        maximumNumberOfIterations = 10000;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment VOWEL";
//        dataSet = importLIBSVMDataSet(workingDirectory, false, 0.0, 0.0);

//        // SEGMENT Data Set Experiment
//        logger.info("Running SEGMENT experiment...");
//        numberOfExperimentRepetitions = 10;
//        numberOfExamplesToPickPerIteration = 1;
//        maximumNumberOfIterations = 1000;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment SEGMENT";
//        dataSet = importLIBSVMDataSet(workingDirectory, false, 0.0, 0.0);

//        // PROTEIN Data Set Experiment
//        logger.info("Running PROTEIN experiment...");
//        numberOfExperimentRepetitions = 5;
//        numberOfExamplesToPickPerIteration = 1;
//        maximumNumberOfIterations = 1000;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment PROTEIN";
//        dataSet = importLIBSVMDataSet(workingDirectory, true, 0.0, 0.0);

//        // COVTYPE Data Set Experiment
//        logger.info("Running COVTYPE experiment...");
//        numberOfExperimentRepetitions = 5;
//        initialNumberOfExamples = 10000;
//        initialRatioOfPositiveExamples = 0.1;
//        numberOfExamplesToPickPerIteration = 10000;
//        String workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment COVTYPE";
//        ImportedDataSet covTypeDataSet = importLIBSVMDataSet(workingDirectory, true);
//        runExperiments(numberOfExperimentRepetitions,
//                       initialNumberOfExamples,
//                       initialRatioOfPositiveExamples,
//                       numberOfExamplesToPickPerIteration,
//                       workingDirectory,
//                       activeLearningMethods,
//                       examplePickingMethod,
//                       covTypeDataSet,
//                       resultTypes);
//        logger.info("Finished all experiments!");

//        // NEWS20 Data Set Experiment
//        logger.info("Running NEWS20 experiment...");
//        numberOfExperimentRepetitions = 1;
//        initialNumberOfExamples = 1000;
//        initialRatioOfPositiveExamples = 0.3;
//        numberOfExamplesToPickPerIteration = 10000;
//        String workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment NEWS20";
//        ImportedDataSet news20TypeDataSet = importLIBSVMDataSet(workingDirectory, true);
//        runExperiments(numberOfExperimentRepetitions,
//                       initialNumberOfExamples,
//                       initialRatioOfPositiveExamples,
//                       numberOfExamplesToPickPerIteration,
//                       workingDirectory,
//                       activeLearningMethods,
//                       examplePickingMethod,
//                       news20TypeDataSet,
//                       resultTypes);
//        logger.info("Finished all experiments!");

//        // ISOLET Data Set Experiment
//        logger.info("Running ISOLET experiment...");
//        numberOfExperimentRepetitions = 5;
//        initialNumberOfExamples = 1000;
//        initialRatioOfPositiveExamples = 0.04;
//        numberOfExamplesToPickPerIteration = 500;
//        String workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment ISOLET";
//        ImportedDataSet isoletDataSet = importISOLETDataSet(workingDirectory);
//        runExperiments(numberOfExperimentRepetitions,
//                       initialNumberOfExamples,
//                       initialRatioOfPositiveExamples,
//                       numberOfExamplesToPickPerIteration,
//                       workingDirectory,
//                       activeLearningMethods,
//                       examplePickingMethod,
//                       isoletDataSet,
//                       resultTypes);

        dataSet.exportStatistics(workingDirectory + "/statistics.txt");
        results = runExperiments(numberOfExperimentRepetitions,
                                 numberOfExamplesToPickPerIteration,
                                 maximumNumberOfIterations,
                                 workingDirectory,
                                 activeLearningMethods,
                                 examplePickingMethod,
                                 dataSet,
                                 resultTypes);
        ConstrainedLearningWithoutReTraining.exportResults(
                numberOfExperimentRepetitions,
                numberOfExamplesToPickPerIteration,
                maximumNumberOfIterations,
                results,
                workingDirectory + "/results.m",
                resultTypes,
                includeTitleInResultsPlot,
                includeHorizontalAxisLabelInResultsPlot,
                includeVerticalAxisLabelInResultsPlot,
                includeLegendInResultsPlot);
        logger.info("Finished all experiments!");
    }

    private static String shadedErrorBarMatlabCode =
            "function varargout=shadedErrorBar(x,y,errBar,lineProps,transparent)\n" +
                    "% function H=shadedErrorBar(x,y,errBar,lineProps,transparent)\n" +
                    "%\n" +
                    "% Purpose \n" +
                    "% Makes a 2-d line plot with a pretty shaded error bar made\n" +
                    "% using patch. Error bar color is chosen automatically.\n" +
                    "%\n" +
                    "% Inputs\n" +
                    "% x - vector of x values [optional, can be left empty]\n" +
                    "% y - vector of y values or a matrix of n observations by m cases\n" +
                    "%     where m has length(x);\n" +
                    "% errBar - if a vector we draw symmetric errorbars. If it has a size\n" +
                    "%          of [2,length(x)] then we draw asymmetric error bars with\n" +
                    "%          row 1 being the upper bar and row 2 being the lower bar\n" +
                    "%          (with respect to y). ** alternatively ** errBar can be a\n" +
                    "%          cellArray of two function handles. The first defines which\n" +
                    "%          statistic the line should be and the second defines the\n" +
                    "%          error bar.\n" +
                    "% lineProps - [optional,'-k' by default] defines the properties of\n" +
                    "%             the data line. e.g.:    \n" +
                    "%             'or-', or {'-or','markerfacecolor',[1,0.2,0.2]}\n" +
                    "% transparent - [optional, 0 by default] if ==1 the shaded error\n" +
                    "%               bar is made transparent, which forces the renderer\n" +
                    "%               to be openGl. However, if this is saved as .eps the\n" +
                    "%               resulting file will contain a raster not a vector\n" +
                    "%               image. \n" +
                    "%\n" +
                    "% Outputs\n" +
                    "% H - a structure of handles to the generated plot objects.     \n" +
                    "%\n" +
                    "%\n" +
                    "% Examples\n" +
                    "% y=randn(30,80); x=1:size(y,2);\n" +
                    "% shadedErrorBar(x,mean(y,1),std(y),'g');\n" +
                    "% shadedErrorBar(x,y,{@median,@std},{'r-o','markerfacecolor','r'});    \n" +
                    "% shadedErrorBar([],y,{@median,@std},{'r-o','markerfacecolor','r'});    \n" +
                    "%\n" +
                    "% Overlay two transparent lines\n" +
                    "% y=randn(30,80)*10; x=(1:size(y,2))-40;\n" +
                    "% shadedErrorBar(x,y,{@mean,@std},'-r',1); \n" +
                    "% hold on\n" +
                    "% y=ones(30,1)*x; y=y+0.06*y.^2+randn(size(y))*10;\n" +
                    "% shadedErrorBar(x,y,{@mean,@std},'-b',1); \n" +
                    "% hold off\n" +
                    "%\n" +
                    "%\n" +
                    "% Rob Campbell - November 2009\n" +
                    "\n" +
                    "\n" +
                    "    \n" +
                    "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    \n" +
                    "% Error checking    \n" +
                    "error(nargchk(3,5,nargin))\n" +
                    "\n" +
                    "\n" +
                    "%Process y using function handles if needed to make the error bar\n" +
                    "%dynamically\n" +
                    "if iscell(errBar) \n" +
                    "    fun1=errBar{1};\n" +
                    "    fun2=errBar{2};\n" +
                    "    errBar=fun2(y);\n" +
                    "    if ~any(errBar(:))\n" +
                    "        errBar = repmat(1e-6 * abs(y), [2, 1]);\n" +
                    "    end\n" +
                    "    y=fun1(y);\n" +
                    "else\n" +
                    "    y=y(:)';\n" +
                    "end\n" +
                    "\n" +
                    "if isempty(x)\n" +
                    "    x=1:length(y);\n" +
                    "else\n" +
                    "    x=x(:)';\n" +
                    "end\n" +
                    "\n" +
                    "\n" +
                    "%Make upper and lower error bars if only one was specified\n" +
                    "if length(errBar)==length(errBar(:))\n" +
                    "    errBar=repmat(errBar(:)',2,1);\n" +
                    "else\n" +
                    "    s=size(errBar);\n" +
                    "    f=find(s==2);\n" +
                    "    if isempty(f), error('errBar has the wrong size'), end\n" +
                    "    if f==2, errBar=errBar'; end\n" +
                    "end\n" +
                    "\n" +
                    "if length(x) ~= length(errBar)\n" +
                    "    error('length(x) must equal length(errBar)')\n" +
                    "end\n" +
                    "\n" +
                    "%Set default options\n" +
                    "defaultProps={'-k'};\n" +
                    "if nargin<4, lineProps=defaultProps; end\n" +
                    "if isempty(lineProps), lineProps=defaultProps; end\n" +
                    "if ~iscell(lineProps), lineProps={lineProps}; end\n" +
                    "\n" +
                    "if nargin<5, transparent=0; end\n" +
                    "\n" +
                    "\n" +
                    "\n" +
                    "\n" +
                    "\n" +
                    "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    \n" +
                    "% Plot to get the parameters of the line \n" +
                    "H.mainLine=plot(x,y,lineProps{:});\n" +
                    "\n" +
                    "\n" +
                    "% Work out the color of the shaded region and associated lines\n" +
                    "% Using alpha requires the render to be openGL and so you can't\n" +
                    "% save a vector image. On the other hand, you need alpha if you're\n" +
                    "% overlaying lines. There we have the option of choosing alpha or a\n" +
                    "% de-saturated solid colour for the patch surface .\n" +
                    "\n" +
                    "col=get(H.mainLine,'color');\n" +
                    "edgeColor=col+(1-col)*0.55;\n" +
                    "patchSaturation=0.15; %How de-saturated or transparent to make patch\n" +
                    "if transparent\n" +
                    "    faceAlpha=patchSaturation;\n" +
                    "    patchColor=col;\n" +
                    "    set(gcf,'renderer','openGL')\n" +
                    "else\n" +
                    "    faceAlpha=1;\n" +
                    "    patchColor=col+(1-col)*(1-patchSaturation);\n" +
                    "    set(gcf,'renderer','painters')\n" +
                    "end\n" +
                    "\n" +
                    "    \n" +
                    "%Calculate the error bars\n" +
                    "uE=y+errBar(1,:);\n" +
                    "lE=y-errBar(2,:);\n" +
                    "\n" +
                    "\n" +
                    "%Add the patch error bar\n" +
                    "holdStatus=ishold;\n" +
                    "if ~holdStatus, hold on,  end\n" +
                    "\n" +
                    "\n" +
                    "%Make the patch\n" +
                    "yP=[lE,fliplr(uE)];\n" +
                    "xP=[x,fliplr(x)];\n" +
                    "\n" +
                    "%remove nans otherwise patch won't work\n" +
                    "xP(isnan(yP))=[];\n" +
                    "yP(isnan(yP))=[];\n" +
                    "\n" +
                    "\n" +
                    "H.patch=patch(xP,yP,1,'facecolor',patchColor,...\n" +
                    "              'edgecolor','none',...\n" +
                    "              'facealpha',faceAlpha);\n" +
                    "\n" +
                    "\n" +
                    "%Make pretty edges around the patch. \n" +
                    "H.edge(1)=plot(x,lE,'-','color',edgeColor);\n" +
                    "H.edge(2)=plot(x,uE,'-','color',edgeColor);\n" +
                    "\n" +
                    "%Now replace the line (this avoids having to bugger about with z coordinates)\n" +
                    "delete(H.mainLine)\n" +
                    "H.mainLine=plot(x,y,lineProps{:});\n" +
                    "\n" +
                    "\n" +
                    "if ~holdStatus, hold off, end\n" +
                    "\n" +
                    "\n" +
                    "if nargout==1\n" +
                    "    varargout{1}=H;\n" +
                    "end\n" +
                    "\n" +
                    "end\n" +
                    "\n" +
                    "function [YY, I, Y0, LB, UB, ADX, NO] = hampel(X, Y, DX, T, varargin)\n" +
                    "% HAMPEL    Hampel Filter.\n" +
                    "%   HAMPEL(X,Y,DX,T,varargin) returns the Hampel filtered values of the \n" +
                    "%   elements in Y. It was developed to detect outliers in a time series, \n" +
                    "%   but it can also be used as an alternative to the standard median \n" +
                    "%   filter.\n" +
                    "%\n" +
                    "%   References\n" +
                    "%   Chapters 1.4.2, 3.2.2 and 4.3.4 in Mining Imperfect Data: Dealing with \n" +
                    "%   Contamination and Incomplete Records by Ronald K. Pearson.\n" +
                    "%\n" +
                    "%   Acknowledgements\n" +
                    "%   I would like to thank Ronald K. Pearson for the introduction to moving\n" +
                    "%   window filters. Please visit his blog at:\n" +
                    "%   http://exploringdatablog.blogspot.com/2012/01/moving-window-filters-and\n" +
                    "%   -pracma.html\n" +
                    "%\n" +
                    "%   X,Y are row or column vectors with an equal number of elements.\n" +
                    "%   The elements in Y should be Gaussian distributed.\n" +
                    "%\n" +
                    "%   Input DX,T,varargin must not contain NaN values!\n" +
                    "%\n" +
                    "%   DX,T are optional scalar values.\n" +
                    "%   DX is a scalar which defines the half width of the filter window. \n" +
                    "%   It is required that DX > 0 and DX should be dimensionally equivalent to\n" +
                    "%   the values in X.\n" +
                    "%   T is a scalar which defines the threshold value used in the equation\n" +
                    "%   |Y - Y0| > T*S0.\n" +
                    "%\n" +
                    "%   Standard Parameters for DX and T:\n" +
                    "%   DX  = 3*median(X(2:end)-X(1:end-1)); \n" +
                    "%   T   = 3;\n" +
                    "%\n" +
                    "%   varargin covers addtional optional input. The optional input must be in\n" +
                    "%   the form of 'PropertyName', PropertyValue.\n" +
                    "%   Supported PropertyNames: \n" +
                    "%   'standard': Use the standard Hampel filter. \n" +
                    "%   'adaptive': Use an experimental adaptive Hampel filter. Explained under\n" +
                    "%   Revision 1 details below.\n" +
                    "% \n" +
                    "%   Supported PropertyValues: Scalar value which defines the tolerance of\n" +
                    "%   the adaptive filter. In the case of standard Hampel filter this value \n" +
                    "%   is ignored.\n" +
                    "%\n" +
                    "%   Output YY,I,Y0,LB,UB,ADX are column vectors containing Hampel filtered\n" +
                    "%   values of Y, a logical index of the replaced values, nominal data,\n" +
                    "%   lower and upper bounds on the Hampel filter and the relative half size \n" +
                    "%   of the local window, respectively.\n" +
                    "%\n" +
                    "%   NO is a scalar that specifies the Number of Outliers detected.\n" +
                    "%\n" +
                    "%   Examples\n" +
                    "%   1. Hampel filter removal of outliers\n" +
                    "%       X           = 1:1000;                           % Pseudo Time\n" +
                    "%       Y           = 5000 + randn(1000, 1);            % Pseudo Data\n" +
                    "%       Outliers    = randi(1000, 10, 1);               % Index of Outliers\n" +
                    "%       Y(Outliers) = Y(Outliers) + randi(1000, 10, 1); % Pseudo Outliers\n" +
                    "%       [YY,I,Y0,LB,UB] = hampel(X,Y);\n" +
                    "%\n" +
                    "%       plot(X, Y, 'b.'); hold on;      % Original Data\n" +
                    "%       plot(X, YY, 'r');               % Hampel Filtered Data\n" +
                    "%       plot(X, Y0, 'b--');             % Nominal Data\n" +
                    "%       plot(X, LB, 'r--');             % Lower Bounds on Hampel Filter\n" +
                    "%       plot(X, UB, 'r--');             % Upper Bounds on Hampel Filter\n" +
                    "%       plot(X(I), Y(I), 'ks');         % Identified Outliers\n" +
                    "%\n" +
                    "%   2. Adaptive Hampel filter removal of outliers\n" +
                    "%       DX          = 1;                                % Window Half size\n" +
                    "%       T           = 3;                                % Threshold\n" +
                    "%       Threshold   = 0.1;                              % AdaptiveThreshold\n" +
                    "%       X           = 1:DX:1000;                        % Pseudo Time\n" +
                    "%       Y           = 5000 + randn(1000, 1);            % Pseudo Data\n" +
                    "%       Outliers    = randi(1000, 10, 1);               % Index of Outliers\n" +
                    "%       Y(Outliers) = Y(Outliers) + randi(1000, 10, 1); % Pseudo Outliers\n" +
                    "%       [YY,I,Y0,LB,UB] = hampel(X,Y,DX,T,'Adaptive',Threshold);\n" +
                    "%\n" +
                    "%       plot(X, Y, 'b.'); hold on;      % Original Data\n" +
                    "%       plot(X, YY, 'r');               % Hampel Filtered Data\n" +
                    "%       plot(X, Y0, 'b--');             % Nominal Data\n" +
                    "%       plot(X, LB, 'r--');             % Lower Bounds on Hampel Filter\n" +
                    "%       plot(X, UB, 'r--');             % Upper Bounds on Hampel Filter\n" +
                    "%       plot(X(I), Y(I), 'ks');         % Identified Outliers\n" +
                    "%\n" +
                    "%   3. Median Filter Based on Filter Window\n" +
                    "%       DX        = 3;                        % Filter Half Size\n" +
                    "%       T         = 0;                        % Threshold\n" +
                    "%       X         = 1:1000;                   % Pseudo Time\n" +
                    "%       Y         = 5000 + randn(1000, 1);    % Pseudo Data\n" +
                    "%       [YY,I,Y0] = hampel(X,Y,DX,T);\n" +
                    "%\n" +
                    "%       plot(X, Y, 'b.'); hold on;    % Original Data\n" +
                    "%       plot(X, Y0, 'r');             % Median Filtered Data\n" +
                    "%\n" +
                    "%   Version: 1.5\n" +
                    "%   Last Update: 09.02.2012\n" +
                    "%\n" +
                    "%   Copyright (c) 2012:\n" +
                    "%   Michael Lindholm Nielsen\n" +
                    "%\n" +
                    "%   --- Revision 5 --- 09.02.2012\n" +
                    "%   (1) Corrected potential error in internal median function.\n" +
                    "%   (2) Removed internal \"keyboard\" command.\n" +
                    "%   (3) Optimized internal Gauss filter.\n" +
                    "%\n" +
                    "%   --- Revision 4 --- 08.02.2012\n" +
                    "%   (1) The elements in X and Y are now temporarily sorted for internal\n" +
                    "%       computations.\n" +
                    "%   (2) Performance optimization.\n" +
                    "%   (3) Added Example 3.\n" +
                    "%\n" +
                    "%   --- Revision 3 --- 06.02.2012\n" +
                    "%   (1) If the number of elements (X,Y) are below 2 the output YY will be a\n" +
                    "%       copy of Y. No outliers will be detected. No error will be issued.\n" +
                    "%\n" +
                    "%   --- Revision 2 --- 05.02.2012\n" +
                    "%   (1) Changed a calculation in the adaptive Hampel filter. The threshold\n" +
                    "%       parameter is now compared to the percentage difference between the\n" +
                    "%       j'th and the j-1 value. Also notice the change from Threshold = 1.1\n" +
                    "%       to Threshold = 0.1 in example 2 above.\n" +
                    "%   (2) Checks if DX,T or varargin contains NaN values.\n" +
                    "%   (3) Now capable of ignoring NaN values in X and Y.\n" +
                    "%   (4) Added output Y0 - Nominal Data.\n" +
                    "%\n" +
                    "%   --- Revision 1 --- 28.01.2012\n" +
                    "%   (1) Replaced output S (Local Scaled Median Absolute Deviation) with\n" +
                    "%       lower (LB) and upper (UB) bounds on the Hampel filter.\n" +
                    "%   (2) Added option to use an experimental adaptive Hampel filter.\n" +
                    "%       The Principle behind this filter is described below.\n" +
                    "%   a) The filter changes the local window size until the change in the \n" +
                    "%       local scaled median absolute deviation is below a threshold value \n" +
                    "%       set by the user. In the above example (2) this parameter is set to \n" +
                    "%       0.1 corresponding to a maximum acceptable change of 10% in the \n" +
                    "%       local scaled median absolute deviation. This process leads to three\n" +
                    "%       locally optimized parameters Y0 (Local Nominal Data Reference \n" +
                    "%       value), S0 (Local Scale of Natural Variation), ADX (Local Adapted \n" +
                    "%       Window half size relative to DX).\n" +
                    "%   b) The optimized parameters are then smoothed by a Gaussian filter with\n" +
                    "%       a standard deviation of DX=2*median(XSort(2:end) - XSort(1:end-1)).\n" +
                    "%       This means that local values are weighted highest, but nearby data \n" +
                    "%       (which should be Gaussian distributed) is also used in refining \n" +
                    "%       ADX, Y0, S0.\n" +
                    "%   \n" +
                    "%   --- Revision 0 --- 26.01.2012\n" +
                    "%   (1) Release of first edition.\n" +
                    "\n" +
                    "%% Error Checking\n" +
                    "% Check for correct number of input arguments\n" +
                    "if nargin < 2\n" +
                    "    error('Not enough input arguments.');\n" +
                    "end\n" +
                    "\n" +
                    "% Check that the number of elements in X match those of Y.\n" +
                    "if ~isequal(numel(X), numel(Y))\n" +
                    "    error('Inputs X and Y must have the same number of elements.');\n" +
                    "end\n" +
                    "\n" +
                    "% Check that X is either a row or column vector\n" +
                    "if size(X, 1) == 1\n" +
                    "    X   = X';   % Change to column vector\n" +
                    "elseif size(X, 2) == 1\n" +
                    "else\n" +
                    "    error('Input X must be either a row or column vector.')\n" +
                    "end\n" +
                    "\n" +
                    "% Check that Y is either a row or column vector\n" +
                    "if size(Y, 1) == 1\n" +
                    "    Y   = Y';   % Change to column vector\n" +
                    "elseif size(Y, 2) == 1\n" +
                    "else\n" +
                    "    error('Input Y must be either a row or column vector.')\n" +
                    "end\n" +
                    "\n" +
                    "% Sort X\n" +
                    "SortX   = sort(X);\n" +
                    "\n" +
                    "% Check that DX is of type scalar\n" +
                    "if exist('DX', 'var')\n" +
                    "    if ~isscalar(DX)\n" +
                    "        error('DX must be a scalar.');\n" +
                    "    elseif DX < 0\n" +
                    "        error('DX must be larger than zero.');\n" +
                    "    end\n" +
                    "else\n" +
                    "    DX  = 3*median(SortX(2:end) - SortX(1:end-1));\n" +
                    "end\n" +
                    "\n" +
                    "% Check that T is of type scalar\n" +
                    "if exist('T', 'var')\n" +
                    "    if ~isscalar(T)\n" +
                    "        error('T must be a scalar.');\n" +
                    "    end\n" +
                    "else\n" +
                    "    T   = 3;\n" +
                    "end\n" +
                    "\n" +
                    "% Check optional input\n" +
                    "if isempty(varargin)\n" +
                    "    Option  = 'standard';\n" +
                    "elseif numel(varargin) < 2\n" +
                    "    error('Optional input must also contain threshold value.');\n" +
                    "else\n" +
                    "    % varargin{1}\n" +
                    "    if ischar(varargin{1})\n" +
                    "        Option      = varargin{1};\n" +
                    "    else\n" +
                    "        error('PropertyName must be of type char.');\n" +
                    "    end\n" +
                    "    % varargin{2}\n" +
                    "    if isscalar(varargin{2})\n" +
                    "        Threshold   = varargin{2};\n" +
                    "    else\n" +
                    "        error('PropertyValue value must be a scalar.');\n" +
                    "    end\n" +
                    "end\n" +
                    "\n" +
                    "% Check that DX,T does not contain NaN values\n" +
                    "if any(isnan(DX) | isnan(T))\n" +
                    "    error('Inputs DX and T must not contain NaN values.');\n" +
                    "end\n" +
                    "\n" +
                    "% Check that varargin does not contain NaN values\n" +
                    "CheckNaN    = cellfun(@isnan, varargin, 'UniformOutput', 0);\n" +
                    "if any(cellfun(@any, CheckNaN))\n" +
                    "    error('Optional inputs must not contain NaN values.');\n" +
                    "end\n" +
                    "\n" +
                    "% Detect/Ignore NaN values in X and Y\n" +
                    "IdxNaN  = isnan(X) | isnan(Y);\n" +
                    "X       = X(~IdxNaN);\n" +
                    "Y       = Y(~IdxNaN);\n" +
                    "\n" +
                    "%% Calculation\n" +
                    "% Preallocation\n" +
                    "YY  = Y;\n" +
                    "I   = false(size(Y));\n" +
                    "S0  = NaN(size(YY));\n" +
                    "Y0  = S0;\n" +
                    "ADX = repmat(DX, size(Y));\n" +
                    "\n" +
                    "if numel(X) > 1\n" +
                    "    switch lower(Option)\n" +
                    "        case 'standard'\n" +
                    "            for i = 1:numel(Y)\n" +
                    "                % Calculate Local Nominal Data Reference value\n" +
                    "                % and Local Scale of Natural Variation\n" +
                    "                [Y0(i), S0(i)]  = localwindow(X, Y, DX, i);\n" +
                    "            end\n" +
                    "        case 'adaptive'\n" +
                    "            % Preallocate\n" +
                    "            Y0Tmp   = S0;\n" +
                    "            S0Tmp   = S0;\n" +
                    "            DXTmp   = (1:numel(S0))'*DX; % Integer variation of Window Half Size\n" +
                    "            \n" +
                    "            % Calculate Initial Guess of Optimal Parameters Y0, S0, ADX\n" +
                    "            for i = 1:numel(Y)\n" +
                    "                % Setup/Reset temporary counter etc.\n" +
                    "                j       = 1;\n" +
                    "                S0Rel   = inf;\n" +
                    "                while S0Rel > Threshold\n" +
                    "                    % Calculate Local Nominal Data Reference value\n" +
                    "                    % and Local Scale of Natural Variation using DXTmp window\n" +
                    "                    [Y0Tmp(j), S0Tmp(j)]    = localwindow(X, Y, DXTmp(j), i);\n" +
                    "                    \n" +
                    "                    % Calculate percent difference relative to previous value\n" +
                    "                    if j > 1\n" +
                    "                        S0Rel   = abs((S0Tmp(j-1) - S0Tmp(j))/(S0Tmp(j-1) + S0Tmp(j))/2);\n" +
                    "                    end\n" +
                    "                    \n" +
                    "                    % Iterate counter\n" +
                    "                    j   = j + 1;\n" +
                    "                end\n" +
                    "                Y0(i)   = Y0Tmp(j - 2);     % Local Nominal Data Reference value\n" +
                    "                S0(i)   = S0Tmp(j - 2);     % Local Scale of Natural Variation\n" +
                    "                ADX(i)  = DXTmp(j - 2)/DX;  % Local Adapted Window size relative to DX\n" +
                    "            end\n" +
                    "            \n" +
                    "            % Gaussian smoothing of relevant parameters\n" +
                    "            DX  = 2*median(SortX(2:end) - SortX(1:end-1));\n" +
                    "            ADX = smgauss(X, ADX, DX);\n" +
                    "            S0  = smgauss(X, S0, DX);\n" +
                    "            Y0  = smgauss(X, Y0, DX);\n" +
                    "        otherwise\n" +
                    "            error('Unknown option ''%s''.', varargin{1});\n" +
                    "    end\n" +
                    "end\n" +
                    "\n" +
                    "%% Prepare Output\n" +
                    "UB      = Y0 + T*S0;            % Save information about local scale\n" +
                    "LB      = Y0 - T*S0;            % Save information about local scale\n" +
                    "Idx     = abs(Y - Y0) > T*S0;   % Index of possible outlier\n" +
                    "YY(Idx) = Y0(Idx);              % Replace outliers with local median value\n" +
                    "I(Idx)  = true;                 % Set Outlier detection\n" +
                    "NO      = sum(I);               % Output number of detected outliers\n" +
                    "\n" +
                    "% Reinsert NaN values detected at error checking stage\n" +
                    "if any(IdxNaN)\n" +
                    "    [YY, I, Y0, LB, UB, ADX]    = rescale(IdxNaN, YY, I, Y0, LB, UB, ADX);\n" +
                    "end\n" +
                    "\n" +
                    "%% Built-in functions\n" +
                    "    function [Y0, S0] = localwindow(X, Y, DX, i)\n" +
                    "        % Index relevant to Local Window\n" +
                    "        Idx = X(i) - DX <= X & X <= X(i) + DX;\n" +
                    "\n" +
                    "        % Calculate Local Nominal Data Reference Value\n" +
                    "        Y0  = median(Y(Idx));\n" +
                    "        \n" +
                    "        % Calculate Local Scale of Natural Variation\n" +
                    "        S0  = 1.4826*median(abs(Y(Idx) - Y0));\n" +
                    "    end\n" +
                    "\n" +
                    "    function M = median(YM)\n" +
                    "        % Isolate relevant values in Y\n" +
                    "        YM  = sort(YM);\n" +
                    "        NYM = numel(YM);\n" +
                    "        \n" +
                    "        % Calculate median\n" +
                    "        if mod(NYM,2)   % Uneven\n" +
                    "            M   = YM((NYM + 1)/2);\n" +
                    "        else            % Even\n" +
                    "            M   = (YM(NYM/2)+YM(NYM/2+1))/2;\n" +
                    "        end\n" +
                    "    end\n" +
                    "\n" +
                    "    function G = smgauss(X, V, DX)\n" +
                    "        % Prepare Xj and Xk\n" +
                    "        Xj  = repmat(X', numel(X), 1);\n" +
                    "        Xk  = repmat(X, 1, numel(X));\n" +
                    "        \n" +
                    "        % Calculate Gaussian weight\n" +
                    "        Wjk = exp(-((Xj - Xk)/(2*DX)).^2);\n" +
                    "        \n" +
                    "        % Calculate Gaussian Filter\n" +
                    "        G   = Wjk*V./sum(Wjk,1)';\n" +
                    "    end\n" +
                    "\n" +
                    "    function varargout = rescale(IdxNaN, varargin)\n" +
                    "        % Output Rescaled Elements\n" +
                    "        varargout    = cell(nargout, 1);\n" +
                    "        for k = 1:nargout\n" +
                    "            Element     = varargin{k};\n" +
                    "            \n" +
                    "            if islogical(Element)\n" +
                    "                ScaledElement   = false(size(IdxNaN));\n" +
                    "            elseif isnumeric(Element)\n" +
                    "                ScaledElement   = NaN(size(IdxNaN));\n" +
                    "            end\n" +
                    "            \n" +
                    "            ScaledElement(~IdxNaN)  = Element;\n" +
                    "            varargout(k)            = {ScaledElement};\n" +
                    "        end\n" +
                    "    end\n" +
                    "end";
}
