package org.platanios.experiment.classification.active;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.experiment.Utilities;
import org.platanios.learn.classification.Label;
import org.platanios.learn.classification.LogisticRegressionAdaGrad;
import org.platanios.learn.classification.TrainableClassifier;
import org.platanios.learn.classification.active.*;
import org.platanios.learn.classification.constraint.Constraint;
import org.platanios.learn.classification.constraint.ConstraintSet;
import org.platanios.learn.classification.constraint.MutualExclusionConstraint;
import org.platanios.learn.classification.constraint.SubsumptionConstraint;
import org.platanios.learn.classification.reflection.LogicIntegrator;
import org.platanios.learn.data.DataInstance;
import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.DataSetInMemory;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.evaluation.PrecisionRecall;
import org.platanios.learn.math.matrix.DenseVector;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ConstrainedLearningWithoutReTraining {
    private static final Logger logger = LogManager.getLogger("Classification / Active / Constrained Learning Experiment");

    private static final Map<ScoringFunction, String> matlabPlotColorsMap = new HashMap<>();

    static {
        matlabPlotColorsMap.put(new RandomScoringFunction(), "[0, 0.4470, 0.7410, 0.8]");
        matlabPlotColorsMap.put(new RandomScoringFunction(true), "[0, 0.7470, 1, 0.8]");
        matlabPlotColorsMap.put(new EntropyScoringFunction(), "[0.8500, 0.3250, 0.0980, 0.8]");
        matlabPlotColorsMap.put(new EntropyScoringFunction(true), "[1, 0.7250, 0.1980, 0.8]");
        matlabPlotColorsMap.put(new ConstraintPropagationScoringFunction(true), "[0.9290, 0.6940, 0.1250, 0.8]");
        matlabPlotColorsMap.put(new ConstraintPropagationScoringFunction(SurpriseFunction.NEGATIVE_LOGARITHM, false), "[0.3010, 0.7450, 0.9330, 0.8]");
        matlabPlotColorsMap.put(new ConstraintPropagationScoringFunction(SurpriseFunction.ONE_MINUS_PROBABILITY, false), "[0.6350, 0.0780, 0.1840, 0.8]");
        matlabPlotColorsMap.put(new ConstraintPropagationScoringFunction(SurpriseFunction.NEGATIVE_LOGARITHM, true), "[0.4940, 0.1840, 0.5560, 0.8]");
        matlabPlotColorsMap.put(new ConstraintPropagationScoringFunction(SurpriseFunction.ONE_MINUS_PROBABILITY, true), "[0.4660, 0.6740, 0.1880, 0.8]");
    }

    private final int numberOfExamplesToPickPerIteration;
    private final int maximumNumberOfIterations;
    private final ScoringFunction scoringFunction;
    private final ExamplePickingMethod examplePickingMethod;
    private final Set<Label> labels;
    private final Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet;
    private final Map<DataInstance<Vector>, Map<Label, Boolean>> trainingDataSet;
    private final Set<ResultType> resultTypes;
    private final ConstraintSet constraints;

    private Map<DataInstance<Vector>, Map<Label, Double>> predictedDataSet;
    private Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> predictedDataSetInstances;

    private ConstrainedLearningWithoutReTraining(int numberOfExamplesToPickPerIteration,
                                                 int maximumNumberOfIterations,
                                                 ScoringFunction scoringFunction,
                                                 ExamplePickingMethod examplePickingMethod,
                                                 String workingDirectory,
                                                 ImportedDataSet importedDataSet,
                                                 Set<ResultType> resultTypes) {
        this.numberOfExamplesToPickPerIteration = numberOfExamplesToPickPerIteration;
        this.maximumNumberOfIterations = maximumNumberOfIterations;
        this.scoringFunction = scoringFunction;
        this.examplePickingMethod = examplePickingMethod;
        this.labels = importedDataSet.labels;
        this.dataSet = copyDataSetMap(importedDataSet.evaluationDataSet);                       // TODO: Temporary fix.
        this.trainingDataSet = copyDataSetMap(importedDataSet.dataSet);
        this.predictedDataSet = copyDataSetMap(importedDataSet.evaluationPredictedDataSet);     // TODO: Temporary fix.
        this.predictedDataSetInstances = importedDataSet.evaluationPredictedDataSetInstances;   // TODO: Temporary fix.
        this.resultTypes = resultTypes;
        this.constraints = importedDataSet.constraints;
    }

    private <T> Map<DataInstance<Vector>, Map<Label, T>> copyDataSetMap(Map<DataInstance<Vector>, Map<Label, T>> map) {
        Map<DataInstance<Vector>, Map<Label, T>> newMap = new HashMap<>();
        for (Map.Entry<DataInstance<Vector>, Map<Label, T>> mapEntry : map.entrySet())
            newMap.put(mapEntry.getKey(), new HashMap<>(mapEntry.getValue()));
        return newMap;
    }

    private ExperimentResults runExperiment() {
        logger.info("Running experiment...");
        Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> classifierDataSet = new HashMap<>();
        Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> testingDataSet = new HashMap<>();
        for (Label label : labels) {
            DataSet<PredictedDataInstance<Vector, Double>> classifierDataSetCopy = new DataSetInMemory<>();
            DataSet<PredictedDataInstance<Vector, Double>> testingDataSetCopy = new DataSetInMemory<>();
            for (PredictedDataInstance<Vector, Double> instance : predictedDataSetInstances.get(label)) {
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
        final long experimentStartTime = System.currentTimeMillis();
        ExperimentResults results = new ExperimentResults();
        Map<DataInstance<Vector>, Map<Label, Boolean>> activeLearningDataSet = new HashMap<>();
        for (DataInstance<Vector> instance : dataSet.keySet())
            activeLearningDataSet.put(instance, new HashMap<>());
        Learning learning;
        if (scoringFunction.propagateConstraints())
            learning =
                    new ConstrainedLearning.Builder(activeLearningDataSet,
                                                    instanceToLabel -> {
                                                        if (predictedDataSet.containsKey(instanceToLabel.getInstance())
                                                                && predictedDataSet.get(instanceToLabel.getInstance()).containsKey(instanceToLabel.getLabel()))
                                                            return predictedDataSet
                                                                    .get(instanceToLabel.getInstance())
                                                                    .get(instanceToLabel.getLabel());
                                                        else
                                                            return 0.0;
                                                    })
                            .activeLearningMethod(scoringFunction)
                            .addConstraints(constraints.getConstraints())
                            .addLabels(labels)
                            .build();
        else
            learning =
                    new Learning.Builder(activeLearningDataSet,
                                         instanceToLabel -> {
                                             if (predictedDataSet.containsKey(instanceToLabel.getInstance())
                                                     && predictedDataSet.get(instanceToLabel.getInstance()).containsKey(instanceToLabel.getLabel()))
                                                 return predictedDataSet
                                                         .get(instanceToLabel.getInstance())
                                                         .get(instanceToLabel.getLabel());
                                             else
                                                 return 0.0;
                                         })
                            .activeLearningMethod(scoringFunction)
                            .addLabels(labels).build();
        for (Map.Entry<Label, DataSet<PredictedDataInstance<Vector, Double>>> instanceEntry : classifierDataSet.entrySet())
            for (PredictedDataInstance<Vector, Double> predictedInstance : instanceEntry.getValue())
                learning.addInstanceToLabel(new DataInstance<>(predictedInstance.name(), predictedInstance.features()),
                                            instanceEntry.getKey());
        // Use logic integrator to make sure all of the constraints are satisfied
        Map<Label, Set<Integer>> labelClassifiers = new HashMap<>();
        int labelIndex = 0;
        for (Label label : labels)
            labelClassifiers.put(label, new HashSet<>(Collections.singletonList(labelIndex++)));
        Map<DataInstance<Vector>, Map<Label, Boolean>> fixedLogicIntegratorDataSet = new HashMap<>();
        for (Map.Entry<DataInstance<Vector>, Map<Label, Boolean>> dataSetEntry : trainingDataSet.entrySet()) {
            DataInstance<Vector> instance = dataSetEntry.getKey();
            if (!fixedLogicIntegratorDataSet.containsKey(instance))
                fixedLogicIntegratorDataSet.put(instance, new HashMap<>());
            for (Map.Entry<Label, Boolean> labelEntry : dataSetEntry.getValue().entrySet()) {
                Label label = labelEntry.getKey();
                fixedLogicIntegratorDataSet.get(instance).put(label, labelEntry.getValue());
            }
        }
        Map<DataInstance<Vector>, Map<Label, Map<Integer, Double>>> logicIntegratorDataSet = new HashMap<>();
        for (Map.Entry<DataInstance<Vector>, Map<Label, Double>> dataSetEntry : predictedDataSet.entrySet()) {
            DataInstance<Vector> instance = dataSetEntry.getKey();
            if (!logicIntegratorDataSet.containsKey(instance))
                logicIntegratorDataSet.put(instance, new HashMap<>());
            for (Map.Entry<Label, Double> labelEntry : dataSetEntry.getValue().entrySet()) {
                Label label = labelEntry.getKey();
                logicIntegratorDataSet.get(instance).put(label, new HashMap<>());
                logicIntegratorDataSet.get(instance).get(label).put(labelClassifiers.get(label).iterator().next(),
                                                                    labelEntry.getValue());
            }
        }
        LogicIntegrator.Builder logicIntegratorBuilder = new LogicIntegrator.Builder(labelClassifiers,
                                                                                     fixedLogicIntegratorDataSet,
                                                                                     logicIntegratorDataSet);
        for (Constraint constraint : constraints.getConstraints())
            if (constraint instanceof MutualExclusionConstraint)
                logicIntegratorBuilder.addConstraint((MutualExclusionConstraint) constraint);
            else if (constraint instanceof SubsumptionConstraint)
                logicIntegratorBuilder.addConstraint((SubsumptionConstraint) constraint);
        LogicIntegrator logicIntegrator = logicIntegratorBuilder.build();
        LogicIntegrator.Output logicIntegratorOutput = logicIntegrator.integratePredictions();
        predictedDataSet = logicIntegratorOutput.getIntegratedDataSet();
        int totalNumberOfExamples = learning.getNumberOfUnlabeledInstances();
        logger.info("Total number of examples: " + totalNumberOfExamples);
        int iterationNumber = 0;
        int numberOfExamplesPicked = 0;
        while (true) {
            // Compute precision-recall curves and related evaluation metrics
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
            if (learning.getNumberOfUnlabeledInstances() == 0 || ++iterationNumber >= maximumNumberOfIterations) {
                if (resultTypes.contains(ResultType.ACTIVE_LEARNING_METHOD_TIMES))
                    results.activeLearningMethodTimesTaken.put(iterationNumber, (long) 0);
                break;
            }
            final long startTime = System.nanoTime();
            switch (examplePickingMethod) {
                case BATCH:
                    List<Learning.InstanceToLabel> selectedInstances =
                            learning.pickInstancesToLabel(numberOfExamplesToPickPerIteration);
                    for (Learning.InstanceToLabel instance : selectedInstances) {
                        boolean trueLabel = dataSet.get(instance.getInstance()).get(instance.getLabel());
                        Map<Label, Boolean> fixedLabels = new HashMap<>(learning.getLabels(instance.getInstance()));
                        int previousNumberOfUnlabeledInstances = learning.getNumberOfUnlabeledInstances();
                        learning.labelInstance(instance, trueLabel);
                        numberOfExamplesPicked += previousNumberOfUnlabeledInstances - learning.getNumberOfUnlabeledInstances();
                        Map<Label, Boolean> newFixedLabels = new HashMap<>(learning.getLabels(instance.getInstance()));
                        fixedLabels.keySet().forEach(newFixedLabels::remove);
                        for (Map.Entry<Label, Boolean> instanceLabelEntry : newFixedLabels.entrySet()) {
                            for (PredictedDataInstance<Vector, Double> predictedInstance : classifierDataSet.get(instanceLabelEntry.getKey()))
                                if (predictedInstance.name().equals(instance.getInstance().name())) {
                                    testingDataSet.get(instanceLabelEntry.getKey()).remove(predictedInstance);
                                    predictedInstance.label(1.0);
                                    predictedInstance.probability(instanceLabelEntry.getValue() ? 1.0 : 0.0);
                                    DataInstance<Vector> dataInstance = new DataInstance<>(predictedInstance.name(), predictedInstance.features());
                                    logicIntegrator.fixDataInstanceLabel(dataInstance, instanceLabelEntry.getKey(), instanceLabelEntry.getValue());
                                    break;
                                }
                        }
                    }
                    // Use logic integrator to make sure all of the constraints are satisfied
                    logicIntegratorOutput = logicIntegrator.integratePredictions();
                    predictedDataSet = logicIntegratorOutput.getIntegratedDataSet();
                    break;
                case PSEUDO_SEQUENTIAL:
                    for (int exampleNumber = 0; exampleNumber < numberOfExamplesToPickPerIteration; exampleNumber++) {
                        Learning.InstanceToLabel instance = learning.pickInstanceToLabel();
                        if (instance == null)
                            break;
                        boolean trueLabel = dataSet.get(instance.getInstance()).get(instance.getLabel());
                        Map<Label, Boolean> fixedLabels = new HashMap<>(learning.getLabels(instance.getInstance()));
                        int previousNumberOfUnlabeledInstances = learning.getNumberOfUnlabeledInstances();
                        learning.labelInstance(instance, trueLabel);
                        numberOfExamplesPicked += previousNumberOfUnlabeledInstances - learning.getNumberOfUnlabeledInstances();
                        Map<Label, Boolean> newFixedLabels = new HashMap<>(learning.getLabels(instance.getInstance()));
                        fixedLabels.keySet().forEach(newFixedLabels::remove);
                        for (Map.Entry<Label, Boolean> instanceLabelEntry : newFixedLabels.entrySet()) {
                            for (PredictedDataInstance<Vector, Double> predictedInstance : classifierDataSet.get(instanceLabelEntry.getKey()))
                                if (predictedInstance.name().equals(instance.getInstance().name())) {
                                    testingDataSet.get(instanceLabelEntry.getKey()).remove(predictedInstance);
                                    predictedInstance.label(1.0);
                                    predictedInstance.probability(instanceLabelEntry.getValue() ? 1.0 : 0.0);
                                    DataInstance<Vector> dataInstance = new DataInstance<>(predictedInstance.name(), predictedInstance.features());
                                    logicIntegrator.fixDataInstanceLabel(dataInstance, instanceLabelEntry.getKey(), instanceLabelEntry.getValue());
                                    break;
                                }
                        }
                    }
                    // Use logic integrator to make sure all of the constraints are satisfied
                    logicIntegratorOutput = logicIntegrator.integratePredictions();
                    predictedDataSet = logicIntegratorOutput.getIntegratedDataSet();
                    break;
                case PSEUDO_SEQUENTIAL_INTEGRATOR:
                    for (int exampleNumber = 0; exampleNumber < numberOfExamplesToPickPerIteration; exampleNumber++) {
                        Learning.InstanceToLabel instance = learning.pickInstanceToLabel();
                        if (instance == null)
                            break;
                        boolean trueLabel = dataSet.get(instance.getInstance()).get(instance.getLabel());
                        Map<Label, Boolean> fixedLabels = new HashMap<>(learning.getLabels(instance.getInstance()));
                        int previousNumberOfUnlabeledInstances = learning.getNumberOfUnlabeledInstances();
                        learning.labelInstance(instance, trueLabel);
                        numberOfExamplesPicked += previousNumberOfUnlabeledInstances - learning.getNumberOfUnlabeledInstances();
                        Map<Label, Boolean> newFixedLabels = new HashMap<>(learning.getLabels(instance.getInstance()));
                        fixedLabels.keySet().forEach(newFixedLabels::remove);
                        for (Map.Entry<Label, Boolean> instanceLabelEntry : newFixedLabels.entrySet()) {
                            for (PredictedDataInstance<Vector, Double> predictedInstance : classifierDataSet.get(instanceLabelEntry.getKey()))
                                if (predictedInstance.name().equals(instance.getInstance().name())) {
                                    testingDataSet.get(instanceLabelEntry.getKey()).remove(predictedInstance);
                                    predictedInstance.label(1.0);
                                    predictedInstance.probability(instanceLabelEntry.getValue() ? 1.0 : 0.0);
                                    DataInstance<Vector> dataInstance = new DataInstance<>(predictedInstance.name(), predictedInstance.features());
                                    logicIntegrator.fixDataInstanceLabel(dataInstance, instanceLabelEntry.getKey(), instanceLabelEntry.getValue());
                                    break;
                                }
                        }
                        // Use logic integrator to make sure all of the constraints are satisfied
                        logicIntegratorOutput = logicIntegrator.integratePredictions();
                        predictedDataSet = logicIntegratorOutput.getIntegratedDataSet();
                    }
                    break;
            }
            final long endTime = System.nanoTime();
            if (resultTypes.contains(ResultType.ACTIVE_LEARNING_METHOD_TIMES))
                results.activeLearningMethodTimesTaken.put(iterationNumber, endTime - startTime);
            logger.info("Completed iteration " + iterationNumber + ": " + numberOfExamplesPicked + "/" + totalNumberOfExamples + " picked...");
        }
        final long experimentEndTime = System.currentTimeMillis();
        results.timeTaken = experimentEndTime - experimentStartTime;
        return results;
    }

    private static ConstraintSet importConstraints(String workingDirectory) {
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
                    String[] childrenLabels = lineParts[1].split(",");
                    for (String childLabel : childrenLabels)
                        constraints.add(new SubsumptionConstraint(new Label(lineParts[0]), new Label(childLabel)));
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
                                      Map<ScoringFunction, List<ExperimentResults>> results,
                                      String filePath,
                                      Set<ResultType> resultTypes,
                                      boolean includeTitle,
                                      boolean includeHorizontalAxisLabel,
                                      boolean includeVerticalAxisLabel,
                                      boolean includeLegend) {
        try {
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
            Map<ScoringFunction, Integer> scoringFunctionIndexMap = new HashMap<>();
            int currentScoringFunctionIndex = 1;
            for (Map.Entry<ScoringFunction, List<ExperimentResults>> resultsEntry : results.entrySet()) {
                scoringFunctionIndexMap.put(resultsEntry.getKey(), currentScoringFunctionIndex++);
                String methodName = resultsEntry.getKey().toString().toLowerCase().replace("-", "_");
                writer.write("% " + resultsEntry.getKey().toString() + "\n");
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
                for (Map.Entry<ScoringFunction, List<ExperimentResults>> resultsEntry : results.entrySet()) {
                    String methodName = resultsEntry.getKey().toString().toLowerCase().replace("-", "_");
                    if (plotIndex == 1)
                        writer.write("p(" + scoringFunctionIndexMap.get(resultsEntry.getKey()) + ") = plot(x, " +
                                             "mean(y_" + methodName + "_" + resultTypePlot.name().toLowerCase() +
                                             ", 1), 'Color', " + matlabPlotColorsMap.get(resultsEntry.getKey()) +
                                             ", 'LineWidth', 3);\n");
                    else
                        writer.write("plot(x, mean(y_" + methodName + "_" + resultTypePlot.name().toLowerCase() +
                                             ", 1), 'Color', " + matlabPlotColorsMap.get(resultsEntry.getKey()) +
                                             ", 'LineWidth', 3);\n");
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
                            writer.write("title('Number of Fixed Labels Per Iteration');\n");
                            break;
                        case ACTIVE_LEARNING_METHOD_TIMES:
                            writer.write("title('Time Spent in Active Learning Method Per Iteration');\n");
                            break;
                    }
//                writer.write("ylim([0 1]);\n");
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
                            writer.write("ylabel('Number of Fixed Labels', 'FontSize', 22);\n");
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
                for (ScoringFunction scoringFunction : results.keySet()) {
                    String methodName = scoringFunction.toString().toLowerCase().replace("-", "_");
                    legendPlotNames.add("p(" + scoringFunctionIndexMap.get(scoringFunction) + ")");
                    legendPlotDescriptions.add("strcat(['" + scoringFunction.toString() + " (' " +
                                                       "num2str(trapz(x, mean(y_" + methodName + "_" +
                                                       ResultType.AVERAGE_AUC_FULL_DATA_SET.name().toLowerCase() +
                                                       ", 1)), '%1.3f') ')'])");
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
        int[] largestIndex = new int[]{0};
        Object[] lastValue = new Object[]{null};
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
        private final ConstraintSet constraints;
        private final Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet;
        private final Map<DataInstance<Vector>, Map<Label, Boolean>> evaluationDataSet;
        private final Map<DataInstance<Vector>, Map<Label, Double>> evaluationPredictedDataSet;
        private final Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> evaluationPredictedDataSetInstances;
        private final Map<Label, DataSetStatistics> statistics;
        private final Map<Label, TrainableClassifier<Vector, Double>> classifiers;

        public ImportedDataSet(Set<Label> labels,
                               ConstraintSet constraints,
                               Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet,
                               Map<DataInstance<Vector>, Map<Label, Boolean>> evaluationDataSet) {
            this(labels, constraints, dataSet, evaluationDataSet, 0.0, 0.0);
        }

        public ImportedDataSet(Set<Label> labels,
                               ConstraintSet constraints,
                               Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet,
                               Map<DataInstance<Vector>, Map<Label, Boolean>> evaluationDataSet,
                               double l1RegularizationWeight,
                               double l2RegularizationWeight) {
            this.labels = labels;
            this.constraints = constraints;
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
            Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> predictedDataSet = new HashMap<>();
            evaluationPredictedDataSetInstances = new ConcurrentHashMap<>();
            classifiers = new HashMap<>();
            for (Label label : labels) {
                predictedDataSet.put(label, new DataSetInMemory<>());
                evaluationPredictedDataSetInstances.put(label, new DataSetInMemory<>());
                Vector randomFeatureVector = dataSet.keySet().iterator().next().features();
                LogisticRegressionAdaGrad.Builder classifierBuilder =
                        new LogisticRegressionAdaGrad.Builder(randomFeatureVector.size())
                                .useBiasTerm(true)
                                .l1RegularizationWeight(l1RegularizationWeight)
                                .l2RegularizationWeight(l2RegularizationWeight)
                                .loggingLevel(1)
                                .sampleWithReplacement(true)
                                .maximumNumberOfIterations(1000)
                                .maximumNumberOfIterationsWithNoPointChange(10)
                                .pointChangeTolerance(1e-8)
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
                    evaluationPredictedDataSetInstances.get(instanceLabelEntry.getKey()).add(predictedInstance);
                }
            labels.parallelStream().forEach(label -> {
                classifiers.get(label).train(predictedDataSet.get(label));
                classifiers.get(label).predictInPlace(evaluationPredictedDataSetInstances.get(label));
                for (PredictedDataInstance<Vector, Double> instance : evaluationPredictedDataSetInstances.get(label))
                    if (instance.label() < 0.5) {
                        instance.label(1 - instance.label());
                        instance.probability(1 - instance.probability());
                    }
            });
            evaluationPredictedDataSet = new ConcurrentHashMap<>();
            for (Map.Entry<Label, DataSet<PredictedDataInstance<Vector, Double>>> instanceEntry : evaluationPredictedDataSetInstances.entrySet()) {
                for (PredictedDataInstance<Vector, Double> predictedInstance : instanceEntry.getValue()) {
                    DataInstance<Vector> instance = new DataInstance<>(predictedInstance.name(),
                                                                       predictedInstance.features());
                    if (!evaluationPredictedDataSet.containsKey(instance))
                        evaluationPredictedDataSet.put(instance, new HashMap<>());
                    evaluationPredictedDataSet.get(instance).put(instanceEntry.getKey(), predictedInstance.probability());
                }
            }
        }

        public Set<Label> getLabels() {
            return labels;
        }

        public Map<DataInstance<Vector>, Map<Label, Boolean>> getDataSet() {
            return dataSet;
        }

        public Map<DataInstance<Vector>, Map<Label, Boolean>> getEvaluationDataSet() {
            return evaluationDataSet;
        }

        public Map<DataInstance<Vector>, Map<Label, Double>> getEvaluationPredictedDataSet() {
            return evaluationPredictedDataSet;
        }

        public Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> getEvaluationPredictedDataSetInstances() {
            return evaluationPredictedDataSetInstances;
        }

        public Map<Label, DataSetStatistics> getStatistics() {
            return statistics;
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
        BATCH,
        PSEUDO_SEQUENTIAL,
        PSEUDO_SEQUENTIAL_INTEGRATOR
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
        Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet = new ConcurrentHashMap<>();
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
        Map<DataInstance<Vector>, Map<Label, Boolean>> evaluationDataSet = new ConcurrentHashMap<>();
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
        return new ImportedDataSet(labels,
                                   importConstraints(workingDirectory),
                                   dataSet,
                                   evaluationDataSet,
                                   l1RegularizationWeight,
                                   l2RegularizationWeight);
    }

    private static ImportedDataSet importLIBSVMDataSet(String workingDirectory,
                                                       boolean sparseFeatures,
                                                       double l1RegularizationWeight,
                                                       double l2RegularizationWeight) {
        logger.info("Importing LIBSVM data set...");
        Map<Vector, String> labeledInstances = new HashMap<>();
        final int[] largestVectorIndex = {0};
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
        Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet = new ConcurrentHashMap<>();
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
        Map<DataInstance<Vector>, Map<Label, Boolean>> evaluationDataSet = new ConcurrentHashMap<>();
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
        return new ImportedDataSet(labels,
                                   importConstraints(workingDirectory),
                                   dataSet,
                                   evaluationDataSet,
                                   l1RegularizationWeight,
                                   l2RegularizationWeight);
    }

    private static ImportedDataSet importNELLDataSet(String cplFeatureMapDirectory,
                                                     String workingDirectory,
                                                     double l1RegularizationWeight,
                                                     double l2RegularizationWeight) {
        logger.info("Importing NELL data set...");
        Map<String, Set<String>> labeledNounPhrases = new HashMap<>();
        try {
            Files.newBufferedReader(Paths.get(workingDirectory + "/labeled_nps.tsv")).lines().forEach(line -> {
                String[] lineParts = line.split("\t");
                if (lineParts.length == 2)
                    labeledNounPhrases.put("LABELED|" + normalizeNounPhrase(lineParts[0]), new HashSet<>(Arrays.asList(lineParts[1].split(","))));
            });
            Files.newBufferedReader(Paths.get(workingDirectory + "/evaluation_nps.tsv")).lines().forEach(line -> {
                String[] lineParts = line.split("\t");
                if (lineParts.length == 2)
                    labeledNounPhrases.put("EVALUATION|" + normalizeNounPhrase(lineParts[0]), new HashSet<>(Arrays.asList(lineParts[1].split(","))));
            });
        } catch (IOException e) {
            throw new IllegalArgumentException("There was a problem with the provided labeled noun phrases file.");
        }
        logger.info("Importing NELL feature map...");
        Map<String, Vector> featureMap;
        if (Files.exists(Paths.get(workingDirectory + "/features.bin")))
            featureMap = Utilities.readMap(workingDirectory + "/features.bin");
        else
            featureMap = buildFeatureMap(cplFeatureMapDirectory,
                                         labeledNounPhrases.keySet()
                                                 .stream()
                                                 .map(nounPhrase -> nounPhrase.split("\\|")[1])
                                                 .collect(Collectors.toSet()));
        Set<Label> labels =
                labeledNounPhrases.values()
                        .stream()
                        .flatMap(Collection::stream)
                        .map(Label::new)
                        .collect(Collectors.toSet());
        Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet = new ConcurrentHashMap<>();
        Map<DataInstance<Vector>, Map<Label, Boolean>> evaluationDataSet = new ConcurrentHashMap<>();
        Set<String> nounPhrasesWithoutFeatures = new HashSet<>();
        Map<String, Set<String>> filteredLabeledNounPhrases = new HashMap<>();
        Map<String, Set<String>> filteredEvaluationNounPhrases = new HashMap<>();
        for (Map.Entry<String, Set<String>> labeledNounPhraseEntry : labeledNounPhrases.entrySet()) {
            String[] nounPhraseParts = labeledNounPhraseEntry.getKey().split("\\|");
            String nounPhraseType = nounPhraseParts[0];
            String nounPhrase = nounPhraseParts[1];
            Set<String> positiveLabels = labeledNounPhraseEntry.getValue();
            Set<String> negativeLabels = labels.stream().map(Label::getName).collect(Collectors.toSet());
            negativeLabels.removeAll(positiveLabels);
            Vector features;
            if (!featureMap.containsKey(nounPhrase)) {
                nounPhrasesWithoutFeatures.add(nounPhrase);
                continue;
            } else {
                if (nounPhraseType.equals("LABELED"))
                    filteredLabeledNounPhrases.put(nounPhrase, positiveLabels);
                else if (nounPhraseType.equals("EVALUATION"))
                    filteredEvaluationNounPhrases.put(nounPhrase, positiveLabels);
                features = featureMap.get(nounPhrase);
            }
            DataInstance<Vector> dataInstance = new DataInstance<>(nounPhrase, features);
            if (nounPhraseType.equals("LABELED")) {
                if (!dataSet.containsKey(dataInstance))
                    dataSet.put(dataInstance, new HashMap<>());
                for (String labelName : positiveLabels)
                    dataSet.get(dataInstance).put(new Label(labelName), true);
                for (String labelName : negativeLabels)
                    dataSet.get(dataInstance).put(new Label(labelName), false);
            } else if (nounPhraseType.equals("EVALUATION")) {
                if (!evaluationDataSet.containsKey(dataInstance))
                    evaluationDataSet.put(dataInstance, new HashMap<>());
                for (String labelName : positiveLabels)
                    evaluationDataSet.get(dataInstance).put(new Label(labelName), true);
                for (String labelName : negativeLabels)
                    evaluationDataSet.get(dataInstance).put(new Label(labelName), false);
            }
        }
//        if (nounPhrasesWithoutFeatures.size() > 0)
//            logger.info("NELL noun phrases without features that were ignored: " + nounPhrasesWithoutFeatures);
//        else
//            logger.info("There were no NELL noun phrases without features in the provided data.");
        exportLabeledNounPhrases(filteredLabeledNounPhrases, workingDirectory + "/filtered_labeled_nps.tsv");
        exportLabeledNounPhrases(filteredEvaluationNounPhrases, workingDirectory + "/filtered_evaluation_nps.tsv");
        return new ImportedDataSet(labels,
                                   importConstraints(workingDirectory),
                                   dataSet,
                                   evaluationDataSet,
                                   l1RegularizationWeight,
                                   l2RegularizationWeight);
    }

    private static String normalizeNounPhrase(String nounPhrase) {
        return nounPhrase.toLowerCase();
    }

    private static Map<String, Vector> buildFeatureMap(String cplFeatureMapDirectory) {
        return buildFeatureMap(cplFeatureMapDirectory, null);
    }

    private static Map<String, Vector> buildFeatureMap(String cplFeatureMapDirectory, Set<String> nounPhrases) {
        Map<String, Vector> featureMap = new HashMap<>();
        Map<String, Integer> contexts;
        try {
            Stream<String> npContextPairsLines = new BufferedReader(new InputStreamReader(new GZIPInputStream(
                    Files.newInputStream(Paths.get(cplFeatureMapDirectory + "/cat_pairs_np-idx.txt.gz"))
            ))).lines();
            Map<String, Map<String, Double>> preprocessedFeatureMap = new HashMap<>();
            npContextPairsLines.forEach(line -> {
                String[] lineParts = line.split("\t");
                String np = lineParts[0];
                if (nounPhrases == null || nounPhrases.contains(np)) {
                    Map<String, Double> contextValues = new HashMap<>();
                    for (int i = 1; i < lineParts.length; i++) {
                        String[] contextParts = lineParts[i].split(" -#- ");
                        contextValues.put(contextParts[0], Double.parseDouble(contextParts[1]));
                    }
                    preprocessedFeatureMap.put(np, contextValues);
                }
            });
            contexts = buildContextsMap(
                    cplFeatureMapDirectory,
                    preprocessedFeatureMap.values()
                            .stream()
                            .map(Map::keySet)
                            .flatMap(Collection::stream)
                            .collect(Collectors.toSet())
            );
            for (Map.Entry<String, Map<String, Double>> preprocessedFeatures : preprocessedFeatureMap.entrySet()) {
                Map<Integer, Double> featuresMap = new TreeMap<>();
                preprocessedFeatures.getValue().entrySet()
                        .stream()
                        .filter(preprocessedFeature -> contexts.containsKey(preprocessedFeature.getKey()))
                        .forEach(preprocessedFeature -> featuresMap.put(contexts.get(preprocessedFeature.getKey()),
                                                                        preprocessedFeature.getValue()));
                featureMap.put(preprocessedFeatures.getKey(), Vectors.sparse(contexts.size(), featuresMap));
            }
        } catch (IOException e) {
            logger.error("An exception was thrown while trying to build the CPL feature map.", e);
        }
        return featureMap;
    }

    private static Map<String, Integer> buildContextsMap(String cplFeatureMapDirectory,
                                                         Set<String> uniqueContexts) throws IOException {
        Map<String, Integer> contexts = new HashMap<>();
        Stream<String> npContextPairsLines = new BufferedReader(new InputStreamReader(new GZIPInputStream(
                Files.newInputStream(Paths.get(cplFeatureMapDirectory + "/cat_contexts.txt.gz"))
        ))).lines();
        int[] contextIndex = { 0 };
        npContextPairsLines.forEachOrdered(line -> {
            String[] lineParts = line.split("\t");
            if (uniqueContexts.contains(lineParts[0]) && !contexts.containsKey(lineParts[0]))
                contexts.put(lineParts[0], contextIndex[0]++);
        });
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

    public static Map<ScoringFunction, List<ExperimentResults>> runExperiments(
            int numberOfExperimentRepetitions,
            int numberOfExamplesToPickPerIteration,
            int maximumNumberOfIterations,
            String workingDirectory,
            ScoringFunction[] scoringFunctions,
            ExamplePickingMethod examplePickingMethod,
            ImportedDataSet importedDataSet,
            Set<ResultType> resultTypes
    ) {
        Map<ScoringFunction, List<ExperimentResults>> results = new ConcurrentHashMap<>();
        Arrays.asList(scoringFunctions).parallelStream().forEach(scoringFunction -> {
            logger.info("Running experiment for " + scoringFunction.toString() + "...");
            results.put(scoringFunction, new ArrayList<>());
            Map<Integer, Double> averageAreasUnderTheCurve = new HashMap<>();
            Map<Integer, Integer> countAreasUnderTheCurve = new HashMap<>();
            int effectiveNumberOfExperimentRepetitions =
                    scoringFunction.equals(new RandomScoringFunction())
                            || scoringFunction.equals(new RandomScoringFunction(true))
                            ? numberOfExperimentRepetitions : 1;
            List<Integer> repetitionsList = new ArrayList<>();
            for (int repetition = 0; repetition < effectiveNumberOfExperimentRepetitions; repetition++)
                repetitionsList.add(repetition);
            repetitionsList.parallelStream().forEach(repetition -> {
                logger.info("Running experiment repetition " + (repetition + 1) + "...");
                ConstrainedLearningWithoutReTraining experiment = new ConstrainedLearningWithoutReTraining(
                        numberOfExamplesToPickPerIteration,
                        maximumNumberOfIterations,
                        scoringFunction,
                        examplePickingMethod,
                        workingDirectory,
                        importedDataSet,
                        resultTypes
                );
                ExperimentResults experimentResults = experiment.runExperiment();
                results.get(scoringFunction).add(experimentResults);
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
            });
        });
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
        boolean includeLegendInResultsPlot = true;
        ScoringFunction[] scoringFunctions = new ScoringFunction[] {
//                new RandomScoringFunction(),
//                new RandomScoringFunction(true),
                new EntropyScoringFunction(),
                new EntropyScoringFunction(true),
                new ConstraintPropagationScoringFunction(true),
//                new ConstraintPropagationScoringFunction(SurpriseFunction.NEGATIVE_LOGARITHM, false),
//                new ConstraintPropagationScoringFunction(SurpriseFunction.ONE_MINUS_PROBABILITY, false),
                new ConstraintPropagationScoringFunction(SurpriseFunction.NEGATIVE_LOGARITHM, true),
                new ConstraintPropagationScoringFunction(SurpriseFunction.ONE_MINUS_PROBABILITY, true)
        };
        ExamplePickingMethod examplePickingMethod = ExamplePickingMethod.PSEUDO_SEQUENTIAL;
        Set<ResultType> resultTypes = new HashSet<>();
        resultTypes.add(ResultType.AVERAGE_AUC_FULL_DATA_SET);
        resultTypes.add(ResultType.NUMBER_OF_EXAMPLES_PICKED);

        String workingDirectory;
        ImportedDataSet dataSet;
        Map<ScoringFunction, List<ExperimentResults>> results;

//        // NELL Data Set Experiment
//        logger.info("Running NELL experiment...");
//        numberOfExperimentRepetitions = 1;
//        numberOfExamplesToPickPerIteration = 300;
//        maximumNumberOfIterations = 100000;
////        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment NELL 2 Class - LogicIteration";
//        workingDirectory = "/home/eplatani/active_learning/data/Experiment NELL 11 Class - LogicIteration";
////        String cplFeatureMapDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/Data Sets/NELL/Server/all-pairs/all-pairs-OC-2011-02-02-smallcontexts50-gz";
//        String cplFeatureMapDirectory = "/nell/data/all-pairs-dir/all-pairs-OC-2011-12-31-big2-gz";
//        dataSet = importNELLDataSet(cplFeatureMapDirectory, workingDirectory, 0.1, 0.1);

//        // IRIS Data Set Experiment
//        logger.info("Running IRIS experiment...");
//        numberOfExperimentRepetitions = 10;
//        numberOfExamplesToPickPerIteration = 1;
//        maximumNumberOfIterations = 1000;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment IRIS - LogicIteration";
//        dataSet = importISOLETDataSet(workingDirectory, 0.0, 0.0);

        // DNA Data Set Experiment
        logger.info("Running DNA experiment...");
        numberOfExperimentRepetitions = 10;
        numberOfExamplesToPickPerIteration = 100;
        maximumNumberOfIterations = 1000;
        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment DNA - LogicIteration";
        dataSet = importLIBSVMDataSet(workingDirectory, true, 0.0, 0.0);

//        // LETTER Data Set Experiment
//        logger.info("Running LETTER experiment...");
//        numberOfExperimentRepetitions = 10;
//        numberOfExamplesToPickPerIteration = 100;
//        maximumNumberOfIterations = 100000;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment LETTER-";
//        dataSet = importLIBSVMDataSet(workingDirectory, false, 0.0, 0.0);

//        // PENDIGITS Data Set Experiment
//        logger.info("Running PENDIGITS experiment...");
//        numberOfExperimentRepetitions = 10;
//        numberOfExamplesToPickPerIteration = 100;
//        maximumNumberOfIterations = 100000;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment PENDIGITS-";
//        dataSet = importLIBSVMDataSet(workingDirectory, false, 0.0, 0.0);

//        // ISOLET Data Set Experiment
//        logger.info("Running ISOLET experiment...");
//        numberOfExperimentRepetitions = 10;
//        numberOfExamplesToPickPerIteration = 100;
//        maximumNumberOfIterations = 100000;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment ISOLET-";
//        dataSet = importISOLETDataSet(workingDirectory, 0.0, 0.0);

//        // PROTEIN Data Set Experiment
//        logger.info("Running PROTEIN experiment...");
//        numberOfExperimentRepetitions = 10;
//        numberOfExamplesToPickPerIteration = 100;
//        maximumNumberOfIterations = 100000;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment PROTEIN - LogicIteration";
//        dataSet = importLIBSVMDataSet(workingDirectory, false, 0.0, 0.0);

//        // SATIMAGE Data Set Experiment
//        logger.info("Running SATIMAGE experiment...");
//        numberOfExperimentRepetitions = 10;
//        numberOfExamplesToPickPerIteration = 100;
//        maximumNumberOfIterations = 100000;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment SATIMAGE-";
//        dataSet = importLIBSVMDataSet(workingDirectory, false, 0.0, 0.0);

//        // VOWEL Data Set Experiment
//        logger.info("Running VOWEL experiment...");
//        numberOfExperimentRepetitions = 10;
//        numberOfExamplesToPickPerIteration = 10;
//        maximumNumberOfIterations = 100000;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment VOWEL - LogicIteration";
//        dataSet = importLIBSVMDataSet(workingDirectory, false, 0.0, 0.0);

//        // SHUTTLE Data Set Experiment
//        logger.info("Running SHUTTLE experiment...");
//        numberOfExperimentRepetitions = 10;
//        numberOfExamplesToPickPerIteration = 100;
//        maximumNumberOfIterations = 100000;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment SHUTTLE-";
//        dataSet = importLIBSVMDataSet(workingDirectory, false, 0.0, 0.0);

//        // SENSIT-VEHICLE Data Set Experiment
//        logger.info("Running SENSIT-VEHICLE experiment...");
//        numberOfExperimentRepetitions = 10;
//        numberOfExamplesToPickPerIteration = 1000;
//        maximumNumberOfIterations = 100000;
//        workingDirectory = "/Users/Anthony/Development/Data Sets/NELL/Active Learning Experiment/Experiment SENSIT-VEHICLE-";
//        dataSet = importLIBSVMDataSet(workingDirectory, false, 0.0, 0.0);

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
                                 scoringFunctions,
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
}
