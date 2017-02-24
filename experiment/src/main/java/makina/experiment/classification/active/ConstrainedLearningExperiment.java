package makina.experiment.classification.active;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import makina.learn.classification.active.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import makina.experiment.Utilities;
import makina.learn.classification.Label;
import makina.learn.classification.LogisticRegressionAdaGrad;
import makina.learn.classification.TrainableClassifier;
import makina.learn.classification.constraint.Constraint;
import makina.learn.classification.constraint.ConstraintSet;
import makina.learn.classification.constraint.MutualExclusionConstraint;
import makina.learn.classification.constraint.SubsumptionConstraint;
import makina.learn.classification.reflection.Integrator;
import makina.learn.classification.reflection.LogicIntegrator;
import makina.learn.data.DataInstance;
import makina.learn.data.DataSet;
import makina.learn.data.DataSetInMemory;
import makina.learn.data.PredictedDataInstance;
import makina.learn.evaluation.BinaryPredictionAccuracy;
import makina.learn.evaluation.PrecisionRecall;
import makina.math.matrix.DenseVector;
import makina.math.matrix.Vector;
import makina.math.matrix.Vectors;

import java.io.*;
import java.lang.reflect.Array;
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
public class ConstrainedLearningExperiment {
    private static final Logger logger = LogManager.getLogger("Classification / Active / Constrained Learning Experiment");

    private static final Map<ScoringFunction, String> matlabPlotColorsMap = new HashMap<>();

    static {
        matlabPlotColorsMap.put(new RandomScoringFunction(), "0.055, 0.1843, 0.4000");
        matlabPlotColorsMap.put(new RandomScoringFunction(true), "0, 0.6000, 0.8500");
        matlabPlotColorsMap.put(new EntropyScoringFunction(), "0, 0.4470, 0.7410");
        matlabPlotColorsMap.put(new EntropyScoringFunction(true), "0, 0.7470, 1");
        matlabPlotColorsMap.put(new ExpectedLabelsFixedScoringFunction(true), "0.9290, 0.6940, 0.1250");
        matlabPlotColorsMap.put(new ConstraintPropagationScoringFunction(true), "0.7500, 0.2250, 0.0480");
//        matlabPlotColorsMap.put(new ConstraintPropagationScoringFunction(SurpriseFunction.NEGATIVE_LOGARITHM, false), "0.3010, 0.7450, 0.9330");
//        matlabPlotColorsMap.put(new ConstraintPropagationScoringFunction(SurpriseFunction.ONE_MINUS_PROBABILITY, false), "0.6350, 0.0780, 0.1840");
        matlabPlotColorsMap.put(new ConstraintPropagationScoringFunction(SurpriseFunction.NEGATIVE_LOGARITHM, true), "0.8500, 0.3250, 0.0980");
        matlabPlotColorsMap.put(new ConstraintPropagationScoringFunction(SurpriseFunction.ONE_MINUS_PROBABILITY, true), "1.000, 0.4850, 0.2280");
    }

    private final BiMap<DataInstance<Vector>, Integer> dataInstances = HashBiMap.create();
    private final Map<Label, TrainableClassifier<Vector, Double>> classifiers = new ConcurrentHashMap<>();
    private final Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> classifiersTrainingDataSet = new HashMap<>();
    private final Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> classifiersTestingDataSet = new ConcurrentHashMap<>();
    private final Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> classifiersEvaluationDataSet = new ConcurrentHashMap<>();
    private final Map<Label, Integer> labelClassifiers = new HashMap<>();

    private final int numberOfExamplesToPickPerIteration;
    private final int maximumNumberOfIterations;
    private final ScoringFunction scoringFunction;
    private final ExamplePickingMethod examplePickingMethod;
    private final boolean useLogicIntegrator;
    private final boolean retrainClassifiers;
    private final Set<Label> labels;
    private final ImportedDataSet importedDataSet;
    private final Map<DataInstance<Vector>, Map<Label, Boolean>> trainingDataSet;
    private final Map<DataInstance<Vector>, Map<Label, Boolean>> testingDataSet;
    private final Map<DataInstance<Vector>, Map<Label, Boolean>> evaluationDataSet;
    private final Set<ResultType> resultTypes;
    private final ConstraintSet constraints;

    private Map<DataInstance<Vector>, Map<Label, Double>> predictedDataSet;
    private LogicIntegrator logicIntegrator;

    private ConstrainedLearningExperiment(int numberOfExamplesToPickPerIteration,
                                          int maximumNumberOfIterations,
                                          ScoringFunction scoringFunction,
                                          ExamplePickingMethod examplePickingMethod,
                                          boolean useLogicIntegrator,
                                          boolean retrainClassifiers,
                                          ImportedDataSet importedDataSet,
                                          Set<ResultType> resultTypes) {
        this.numberOfExamplesToPickPerIteration = numberOfExamplesToPickPerIteration;
        this.maximumNumberOfIterations = maximumNumberOfIterations;
        this.scoringFunction = scoringFunction;
        this.examplePickingMethod = examplePickingMethod;
        this.useLogicIntegrator = useLogicIntegrator;
        this.retrainClassifiers = retrainClassifiers;
        this.labels = importedDataSet.labels;
        this.importedDataSet = importedDataSet;
        this.trainingDataSet = copyDataSetMap(importedDataSet.trainingDataSet);
        Map<DataInstance<Vector>, Map<Label, Boolean>> testingDataSet = copyDataSetMap(importedDataSet.testingDataSet);
        List<DataInstance<Vector>> shuffledTestInstances = shuffledInstances(testingDataSet);
        int testSize = (int) Math.floor(0.95 * shuffledTestInstances.size());
        Set<DataInstance<Vector>> testInstances = new HashSet<>(shuffledTestInstances.subList(0, testSize));
        Set<DataInstance<Vector>> evaluationInstances = new HashSet<>(shuffledTestInstances.subList(testSize, shuffledTestInstances.size()));
        this.testingDataSet = testingDataSet
                .entrySet()
                .stream()
                .filter(p -> testInstances.contains(p.getKey()))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        this.evaluationDataSet = testingDataSet
                .entrySet()
                .stream()
                .filter(p -> evaluationInstances.contains(p.getKey()))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        this.resultTypes = resultTypes;
        this.constraints = importedDataSet.constraints;
        for (Label label : labels) {
            classifiersTrainingDataSet.put(label, new DataSetInMemory<>());
            classifiersTestingDataSet.put(label, new DataSetInMemory<>());
            classifiersEvaluationDataSet.put(label, new DataSetInMemory<>());
            Vector randomFeatureVector = trainingDataSet.keySet().iterator().next().features();
            LogisticRegressionAdaGrad.Builder classifierBuilder =
                    new LogisticRegressionAdaGrad.Builder(randomFeatureVector.size())
                            .useBiasTerm(true)
                            .l1RegularizationWeight(importedDataSet.l1RegularizationWeight)
                            .l2RegularizationWeight(importedDataSet.l2RegularizationWeight)
                            .loggingLevel(0)
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
        for (Map.Entry<DataInstance<Vector>, Map<Label, Boolean>> instanceEntry : this.trainingDataSet.entrySet()) {
            dataInstances.putIfAbsent(instanceEntry.getKey(), dataInstances.size());
            for (Map.Entry<Label, Boolean> instanceLabelEntry : instanceEntry.getValue().entrySet()) {
                PredictedDataInstance<Vector, Double> predictedInstance =
                        new PredictedDataInstance<>(instanceEntry.getKey().name(),
                                                    instanceEntry.getKey().features(),
                                                    instanceLabelEntry.getValue() ? 1.0 : 0.0,
                                                    null,
                                                    1.0);
                classifiersTrainingDataSet.get(instanceLabelEntry.getKey()).add(predictedInstance);
            }
        }
        for (Map.Entry<DataInstance<Vector>, Map<Label, Boolean>> instanceEntry : this.testingDataSet.entrySet()) {
            dataInstances.putIfAbsent(instanceEntry.getKey(), dataInstances.size());
            for (Map.Entry<Label, Boolean> instanceLabelEntry : instanceEntry.getValue().entrySet()) {
                PredictedDataInstance<Vector, Double> predictedInstance =
                        new PredictedDataInstance<>(instanceEntry.getKey().name(),
                                                    instanceEntry.getKey().features(),
                                                    instanceLabelEntry.getValue() ? 1.0 : 0.0,
                                                    null,
                                                    1.0);
                classifiersTestingDataSet.get(instanceLabelEntry.getKey()).add(predictedInstance);
            }
        }
        for (Map.Entry<DataInstance<Vector>, Map<Label, Boolean>> instanceEntry : this.evaluationDataSet.entrySet()) {
            for (Map.Entry<Label, Boolean> instanceLabelEntry : instanceEntry.getValue().entrySet()) {
                PredictedDataInstance<Vector, Double> predictedInstance =
                        new PredictedDataInstance<>(instanceEntry.getKey().name(),
                                instanceEntry.getKey().features(),
                                instanceLabelEntry.getValue() ? 1.0 : 0.0,
                                null,
                                1.0);
                classifiersEvaluationDataSet.get(instanceLabelEntry.getKey()).add(predictedInstance);
            }
        }
    }

    private <T> List<DataInstance<Vector>> shuffledInstances(Map<DataInstance<Vector>, Map<Label, T>> map) {
        List<DataInstance<Vector>> instances = new ArrayList<>(map.keySet());
        Collections.shuffle(instances);
        return instances;
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
        Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> evaluationDataSet = new HashMap<>();
        for (Label label : labels) {
            DataSet<PredictedDataInstance<Vector, Double>> classifierDataSetCopy = new DataSetInMemory<>();
            DataSet<PredictedDataInstance<Vector, Double>> testingDataSetCopy = new DataSetInMemory<>();
            DataSet<PredictedDataInstance<Vector, Double>> evaluationDataSetCopy = new DataSetInMemory<>();
            for (PredictedDataInstance<Vector, Double> instance : classifiersTestingDataSet.get(label)) {
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
            for (PredictedDataInstance<Vector, Double> instance : classifiersEvaluationDataSet.get(label)) {
                evaluationDataSetCopy.add(new PredictedDataInstance<>(instance.name(),
                        instance.features(),
                        instance.label(),
                        instance.source(),
                        instance.probability()));
            }
            classifierDataSet.put(label, classifierDataSetCopy);
            testingDataSet.put(label, testingDataSetCopy);
            evaluationDataSet.put(label, evaluationDataSetCopy);
        }
        trainClassifiers(evaluationDataSet);
        final long experimentStartTime = System.currentTimeMillis();
        ExperimentResults results = new ExperimentResults();
        Map<DataInstance<Vector>, Map<Label, Boolean>> activeLearningDataSet = new HashMap<>();
        for (DataInstance<Vector> instance : this.testingDataSet.keySet())
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
        int labelIndex = 0;
        for (Label label : labels)
            labelClassifiers.put(label, labelIndex++);
        if (useLogicIntegrator) {
            buildLogicIntegrator();
            updatePredictedData(logicIntegrator.integratedData());
        }
        int totalNumberOfExamples = learning.getNumberOfUnlabeledInstances();
        logger.info("Total number of examples: " + totalNumberOfExamples);
        int iterationNumber = 0;
        int numberOfExamplesPicked = 0;
        while (true) {
            // Compute precision-recall curves and related evaluation metrics
            PrecisionRecall<Vector, Double> fullPrecisionRecall = new PrecisionRecall<>(1000);
            PrecisionRecall<Vector, Double> testingPrecisionRecall = new PrecisionRecall<>(1000);
            PrecisionRecall<Vector, Double> evaluationPrecisionRecall = new PrecisionRecall<>(1000);
            BinaryPredictionAccuracy<Vector, Double> fullPredictionAccuracy = new BinaryPredictionAccuracy<>();
            BinaryPredictionAccuracy<Vector, Double> testingPredictionAccuracy = new BinaryPredictionAccuracy<>();
            BinaryPredictionAccuracy<Vector, Double> evaluationPredictionAccuracy = new BinaryPredictionAccuracy<>();
            final double[] weightsSum = { 0.0 };
            final double[] fullAverageAUC = { 0.0 };
            final double[] testingAverageAUC = { 0.0 };
            final double[] evaluationAverageAUC = { 0.0 };
            final double[] fullAveragePredictionAccuracy = { 0.0 };
            final double[] testingAveragePredictionAccuracy = { 0.0 };
            final double[] evaluationAveragePredictionAccuracy = { 0.0 };
            labels.stream().forEach(label -> {
                double weight =
                        importedDataSet.trainingDataSetStatistics.get(label).numberOfPositiveExamples
                                + importedDataSet.testingDataSetStatistics.get(label).numberOfPositiveExamples;
                weightsSum[0] += weight;
                if (resultTypes.contains(ResultType.AVERAGE_AUC_FULL_DATA_SET)) {
                    fullPrecisionRecall.addResult(label.name(),
                            classifierDataSet.get(label),
                            dataInstance -> this.testingDataSet.get(new DataInstance<>(dataInstance.name(),
                                    dataInstance.features())).get(label));
                    fullAverageAUC[0] += weight * fullPrecisionRecall.getAreaUnderCurve(label.name());
                }
                if (resultTypes.contains(ResultType.AVERAGE_AUC_TESTING_DATA_SET)) {
                    testingPrecisionRecall.addResult(label.name(),
                            testingDataSet.get(label),
                            dataInstance -> this.testingDataSet.get(new DataInstance<>(dataInstance.name(),
                                    dataInstance.features())).get(label));
                    testingAverageAUC[0] += weight * testingPrecisionRecall.getAreaUnderCurve(label.name());
                }
                if (resultTypes.contains(ResultType.AVERAGE_AUC_EVALUATION_DATA_SET)) {
                    evaluationPrecisionRecall.addResult(label.name(),
                            evaluationDataSet.get(label),
                            dataInstance -> this.evaluationDataSet.get(new DataInstance<>(dataInstance.name(),
                                    dataInstance.features())).get(label));
                    evaluationAverageAUC[0] += weight * evaluationPrecisionRecall.getAreaUnderCurve(label.name());
                }
                if (resultTypes.contains(ResultType.PREDICTION_ACCURACY_FULL_DATA_SET)) {
                    fullPredictionAccuracy.addResult(label.name(),
                            classifierDataSet.get(label),
                            dataInstance -> this.testingDataSet.get(new DataInstance<>(dataInstance.name(),
                                    dataInstance.features())).get(label));
                    fullAveragePredictionAccuracy[0] += weight * fullPredictionAccuracy.getPredictionAccuracy(label.name());
                }
                if (resultTypes.contains(ResultType.PREDICTION_ACCURACY_TESTING_DATA_SET)) {
                    testingPredictionAccuracy.addResult(label.name(),
                            testingDataSet.get(label),
                            dataInstance -> this.testingDataSet.get(new DataInstance<>(dataInstance.name(),
                                    dataInstance.features())).get(label));
                    testingAveragePredictionAccuracy[0] += weight * testingPredictionAccuracy.getPredictionAccuracy(label.name());
                }
                if (resultTypes.contains(ResultType.PREDICTION_ACCURACY_EVALUATION_DATA_SET)) {
                    evaluationPredictionAccuracy.addResult(label.name(),
                            evaluationDataSet.get(label),
                            dataInstance -> this.evaluationDataSet.get(new DataInstance<>(dataInstance.name(),
                                    dataInstance.features())).get(label));
                    evaluationAveragePredictionAccuracy[0] += weight * evaluationPredictionAccuracy.getPredictionAccuracy(label.name());
                }
            });
            if (resultTypes.contains(ResultType.AVERAGE_AUC_FULL_DATA_SET))
                results.averageAreasUnderTheCurve.put(iterationNumber, fullAverageAUC[0] / weightsSum[0]);
            if (resultTypes.contains(ResultType.AVERAGE_AUC_TESTING_DATA_SET))
                results.averageTestingAreasUnderTheCurve.put(iterationNumber, testingAverageAUC[0] / weightsSum[0]);
            if (resultTypes.contains(ResultType.AVERAGE_AUC_EVALUATION_DATA_SET))
                results.averageEvaluationAreasUnderTheCurve.put(iterationNumber, evaluationAverageAUC[0] / weightsSum[0]);
            if (resultTypes.contains(ResultType.AUC_1_ITERATIONS_FULL_DATA_SET)
                    && fullAverageAUC[0] / weightsSum[0] >= 0.999)
                results.auc1IterationsFull = iterationNumber;
            if (resultTypes.contains(ResultType.AUC_1_ITERATIONS_TESTING_DATA_SET)
                    && testingAverageAUC[0] / weightsSum[0] >= 0.999)
                results.auc1IterationsTesting = iterationNumber;
            if (resultTypes.contains(ResultType.AUC_1_ITERATIONS_EVALUATION_DATA_SET)
                    && evaluationAverageAUC[0] / weightsSum[0] >= 0.999)
                results.auc1IterationsEvaluation = iterationNumber;
            if (resultTypes.contains(ResultType.PREDICTION_ACCURACY_FULL_DATA_SET))
                results.predictionAccuracies.put(iterationNumber, fullAveragePredictionAccuracy[0] / weightsSum[0]);
            if (resultTypes.contains(ResultType.PREDICTION_ACCURACY_TESTING_DATA_SET))
                results.testingPredictionAccuracies.put(iterationNumber, testingAveragePredictionAccuracy[0] / weightsSum[0]);
            if (resultTypes.contains(ResultType.PREDICTION_ACCURACY_EVALUATION_DATA_SET))
                results.evaluationPredictionAccuracies.put(iterationNumber, evaluationAveragePredictionAccuracy[0] / weightsSum[0]);
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
                    for (Learning.InstanceToLabel instance : selectedInstances)
                        numberOfExamplesPicked += labelInstance(learning, instance, classifierDataSet, testingDataSet);
                    if (retrainClassifiers) {
                        trainClassifiers(evaluationDataSet);
                        if (useLogicIntegrator)
                            buildLogicIntegrator();
                    }
                    if (useLogicIntegrator) {
                        updatePredictedData(logicIntegrator.integratedData());
                    }
                    break;
                case PSEUDO_SEQUENTIAL:
                    for (int exampleNumber = 0; exampleNumber < numberOfExamplesToPickPerIteration; exampleNumber++) {
                        Learning.InstanceToLabel instance = learning.pickInstanceToLabel();
                        if (instance == null)
                            break;
                        numberOfExamplesPicked += labelInstance(learning, instance, classifierDataSet, testingDataSet);
                    }
                    if (retrainClassifiers) {
                        trainClassifiers(evaluationDataSet);
                        if (useLogicIntegrator)
                            buildLogicIntegrator();
                    }
                    if (useLogicIntegrator) {
                        updatePredictedData(logicIntegrator.integratedData());
                    }
                    break;
                case PSEUDO_SEQUENTIAL_INTEGRATOR:
                    for (int exampleNumber = 0; exampleNumber < numberOfExamplesToPickPerIteration; exampleNumber++) {
                        Learning.InstanceToLabel instance = learning.pickInstanceToLabel();
                        if (instance == null)
                            break;
                        numberOfExamplesPicked += labelInstance(learning, instance, classifierDataSet, testingDataSet);
                        if (retrainClassifiers) {
                            trainClassifiers(evaluationDataSet);
                            if (useLogicIntegrator)
                                buildLogicIntegrator();
                        }
                        if (useLogicIntegrator) {
                            updatePredictedData(logicIntegrator.integratedData());
                        }
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

    private void trainClassifiers(Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> evaluationDataSet) {
        labels.parallelStream().forEach(label -> {
            classifiers.get(label).train(classifiersTrainingDataSet.get(label));
            classifiers.get(label).predictInPlace(classifiersTestingDataSet.get(label));
            for (PredictedDataInstance<Vector, Double> instance : classifiersTestingDataSet.get(label))
                if (instance.label() < 0.5) {
                    instance.label(1 - instance.label());
                    instance.probability(1 - instance.probability());
                }
            if (evaluationDataSet != null) {
                classifiers.get(label).predictInPlace(evaluationDataSet.get(label));
                for (PredictedDataInstance<Vector, Double> instance : evaluationDataSet.get(label))
                    if (instance.label() < 0.5) {
                        instance.label(1 - instance.label());
                        instance.probability(1 - instance.probability());
                    }
            }
        });
        predictedDataSet = new ConcurrentHashMap<>();
        for (Map.Entry<Label, DataSet<PredictedDataInstance<Vector, Double>>> instanceEntry : classifiersTestingDataSet.entrySet()) {
            for (PredictedDataInstance<Vector, Double> predictedInstance : instanceEntry.getValue()) {
                DataInstance<Vector> instance = new DataInstance<>(predictedInstance.name(),
                                                                   predictedInstance.features());
                if (!predictedDataSet.containsKey(instance))
                    predictedDataSet.put(instance, new HashMap<>());
                predictedDataSet.get(instance).put(instanceEntry.getKey(), predictedInstance.probability());
            }
        }
    }

    private void buildLogicIntegrator() {
        List<Integrator.Data.ObservedInstance> observedInstances = new ArrayList<>();
        for (Map.Entry<DataInstance<Vector>, Map<Label, Boolean>> dataSetEntry : trainingDataSet.entrySet())
            observedInstances.addAll(dataSetEntry.getValue().entrySet().stream()
                                             .map(labelEntry -> new Integrator.Data.ObservedInstance(
                                                     dataInstances.get(dataSetEntry.getKey()),
                                                     labelEntry.getKey(),
                                                     labelEntry.getValue())
                                             ).collect(Collectors.toList()));
        List<Integrator.Data.PredictedInstance> predictedInstances = new ArrayList<>();
        for (Map.Entry<DataInstance<Vector>, Map<Label, Double>> dataSetEntry : predictedDataSet.entrySet())
            predictedInstances.addAll(dataSetEntry.getValue().entrySet().stream()
                                              .map(labelEntry -> new Integrator.Data.PredictedInstance(
                                                      dataInstances.get(dataSetEntry.getKey()),
                                                      labelEntry.getKey(),
                                                      labelClassifiers.get(labelEntry.getKey()),
                                                      labelEntry.getValue())
                                              ).collect(Collectors.toList()));
        LogicIntegrator.Builder logicIntegratorBuilder =
                new LogicIntegrator.Builder(new Integrator.Data<>(predictedInstances),
                                            new Integrator.Data<>(observedInstances));
        for (Constraint constraint : constraints.getConstraints())
            if (constraint instanceof MutualExclusionConstraint)
                logicIntegratorBuilder.addConstraint((MutualExclusionConstraint) constraint);
            else if (constraint instanceof SubsumptionConstraint)
                logicIntegratorBuilder.addConstraint((SubsumptionConstraint) constraint);
        logicIntegrator = logicIntegratorBuilder.build();
    }

    private void updatePredictedData(Integrator.Data<Integrator.Data.PredictedInstance> integratedData) {
        predictedDataSet = new HashMap<>();
        for (Integrator.Data.PredictedInstance instance : integratedData) {
            DataInstance<Vector> dataInstance = dataInstances.inverse().get(instance.id());
            predictedDataSet.computeIfAbsent(dataInstance, key -> new HashMap<>());
            predictedDataSet.get(dataInstance).put(instance.label(), instance.value());
        }
    }

    private int labelInstance(Learning learning,
                              Learning.InstanceToLabel instance,
                              Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> classifierDataSet,
                              Map<Label, DataSet<PredictedDataInstance<Vector, Double>>> testingDataSet) {
        boolean trueLabel = this.testingDataSet.get(instance.getInstance()).get(instance.getLabel());
        Map<Label, Boolean> fixedLabels = new HashMap<>(learning.getLabels(instance.getInstance()));
        int previousNumberOfUnlabeledInstances = learning.getNumberOfUnlabeledInstances();
        learning.labelInstance(instance, trueLabel);
        int numberOfExamplesPicked = previousNumberOfUnlabeledInstances - learning.getNumberOfUnlabeledInstances();
        Map<Label, Boolean> newFixedLabels = new HashMap<>(learning.getLabels(instance.getInstance()));
        fixedLabels.keySet().forEach(newFixedLabels::remove);
        for (Map.Entry<Label, Boolean> instanceLabelEntry : newFixedLabels.entrySet()) {
            for (PredictedDataInstance<Vector, Double> predictedInstance : classifierDataSet.get(instanceLabelEntry.getKey()))
                if (predictedInstance.name().equals(instance.getInstance().name())) {
                    testingDataSet.get(instanceLabelEntry.getKey()).remove(predictedInstance);
                    predictedInstance.label(1.0);
                    predictedInstance.probability(instanceLabelEntry.getValue() ? 1.0 : 0.0);
                    if (retrainClassifiers) {
                        DataInstance<Vector> dataInstance = new DataInstance<>(predictedInstance.name(), predictedInstance.features());
                        if (!trainingDataSet.containsKey(dataInstance))
                            trainingDataSet.put(dataInstance, new HashMap<>());
                        trainingDataSet.get(dataInstance).put(instanceLabelEntry.getKey(), instanceLabelEntry.getValue());
                        predictedDataSet.get(dataInstance).remove(instanceLabelEntry.getKey());
                        classifiersTrainingDataSet.get(instanceLabelEntry.getKey()).add(predictedInstance);
                        Iterator<PredictedDataInstance<Vector, Double>> testingDataSetIterator = classifiersTestingDataSet.get(instanceLabelEntry.getKey()).iterator();
                        while (testingDataSetIterator.hasNext())
                            if (testingDataSetIterator.next().name().equals(dataInstance.name())) {
                                testingDataSetIterator.remove();
                                break;
                            }
                    }
                    if (useLogicIntegrator)
                        logicIntegrator.fixDataInstanceLabel(
                                dataInstances.get(new DataInstance<>(predictedInstance.name(),
                                                                     predictedInstance.features())),
                                instanceLabelEntry.getKey(),
                                instanceLabelEntry.getValue()
                        );
                    break;
                }
        }
        return numberOfExamplesPicked;
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
            File file = new File(filePath);
            file.getParentFile().mkdirs();
            FileWriter writer = new FileWriter(file);
            writer.write("% numberOfExperimentRepetitions = " + numberOfExperimentRepetitions + "\n"
                                 + "% numberOfExamplesToPickPerIteration = " + numberOfExamplesToPickPerIteration + "\n"
                                 + "% maximumNumberOfIterations = " + maximumNumberOfIterations + "\n");
            int largestVectorSize = 0;
            for (ExperimentResults result : results.values().stream().flatMap(Collection::stream).collect(Collectors.toList()))
                largestVectorSize = Math.max(largestVectorSize, result.averageAreasUnderTheCurve.keySet().size());
            for (ExperimentResults result : results.values().stream().flatMap(Collection::stream).collect(Collectors.toList()))
                largestVectorSize = Math.max(largestVectorSize, result.averageTestingAreasUnderTheCurve.keySet().size());
            for (ExperimentResults result : results.values().stream().flatMap(Collection::stream).collect(Collectors.toList()))
                largestVectorSize = Math.max(largestVectorSize, result.averageEvaluationAreasUnderTheCurve.keySet().size());
            StringJoiner xStringJoiner = new StringJoiner(",", "[", "]");
            for (int xIndex = 0; xIndex < largestVectorSize; xIndex++)
                xStringJoiner.add(String.valueOf(xIndex));
            writer.write("x = " + xStringJoiner.toString() + ";\n");
            Map<ScoringFunction, Integer> scoringFunctionIndexMap = new HashMap<>();
            int currentScoringFunctionIndex = 1;
            int xLimit = 0;
            double averageAreaUnderCurveYLimitMax = 0;
            double averageAreaUnderCurveYLimitMin = Double.MAX_VALUE;
            double predictionAccuracyYLimitMax = 0;
            double predictionAccuracyYLimitMin = Double.MAX_VALUE;
            double numberOfFixedExamplesYLimitMax = 0;
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
                if (resultTypes.contains(ResultType.PREDICTION_ACCURACY_FULL_DATA_SET)) {
                    writer.write("x_" + methodName + "_" + ResultType.PREDICTION_ACCURACY_FULL_DATA_SET.name().toLowerCase() + " = zeros(" + resultsEntry.getValue().size() + ", " + largestVectorSize + ");\n");
                    writer.write("y_" + methodName + "_" + ResultType.PREDICTION_ACCURACY_FULL_DATA_SET.name().toLowerCase() + " = zeros(" + resultsEntry.getValue().size() + ", " + largestVectorSize + ");\n");
                }
                if (resultTypes.contains(ResultType.PREDICTION_ACCURACY_TESTING_DATA_SET)) {
                    writer.write("x_" + methodName + "_" + ResultType.PREDICTION_ACCURACY_TESTING_DATA_SET.name().toLowerCase() + " = zeros(" + resultsEntry.getValue().size() + ", " + largestVectorSize + ");\n");
                    writer.write("y_" + methodName + "_" + ResultType.PREDICTION_ACCURACY_TESTING_DATA_SET.name().toLowerCase() + " = zeros(" + resultsEntry.getValue().size() + ", " + largestVectorSize + ");\n");
                }
                if (resultTypes.contains(ResultType.PREDICTION_ACCURACY_EVALUATION_DATA_SET)) {
                    writer.write("x_" + methodName + "_" + ResultType.PREDICTION_ACCURACY_EVALUATION_DATA_SET.name().toLowerCase() + " = zeros(" + resultsEntry.getValue().size() + ", " + largestVectorSize + ");\n");
                    writer.write("y_" + methodName + "_" + ResultType.PREDICTION_ACCURACY_EVALUATION_DATA_SET.name().toLowerCase() + " = zeros(" + resultsEntry.getValue().size() + ", " + largestVectorSize + ");\n");
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
                    if (resultTypes.contains(ResultType.AVERAGE_AUC_FULL_DATA_SET)) {
                        writer.write(simpleMapToMatlabString(result.averageAreasUnderTheCurve, methodName, ResultType.AVERAGE_AUC_FULL_DATA_SET.name().toLowerCase(), "1.0", experimentIndex, largestVectorSize) + "\n");
                        averageAreaUnderCurveYLimitMax = Math.max(averageAreaUnderCurveYLimitMax, result.averageAreasUnderTheCurve.values().stream().mapToDouble(x -> x).max().orElse(0.0));
                        averageAreaUnderCurveYLimitMin = Math.min(averageAreaUnderCurveYLimitMin, result.averageAreasUnderTheCurve.values().stream().mapToDouble(x -> x).min().orElse(0.0));
                        xLimit = Math.max(xLimit, result.averageAreasUnderTheCurve.keySet().stream().mapToInt(x -> x).max().orElse(0));
                    }
                    if (resultTypes.contains(ResultType.AVERAGE_AUC_TESTING_DATA_SET)) {
                        writer.write(simpleMapToMatlabString(result.averageTestingAreasUnderTheCurve, methodName, ResultType.AVERAGE_AUC_TESTING_DATA_SET.name().toLowerCase(), "1.0", experimentIndex, largestVectorSize) + "\n");
                        averageAreaUnderCurveYLimitMax = Math.max(averageAreaUnderCurveYLimitMax, result.averageTestingAreasUnderTheCurve.values().stream().mapToDouble(x -> x).max().orElse(0.0));
                        averageAreaUnderCurveYLimitMin = Math.min(averageAreaUnderCurveYLimitMin, result.averageTestingAreasUnderTheCurve.values().stream().mapToDouble(x -> x).min().orElse(0.0));
                        xLimit = Math.max(xLimit, result.averageTestingAreasUnderTheCurve.keySet().stream().mapToInt(x -> x).max().orElse(0));
                    }
                    if (resultTypes.contains(ResultType.AVERAGE_AUC_EVALUATION_DATA_SET)) {
                        writer.write(simpleMapToMatlabString(result.averageEvaluationAreasUnderTheCurve, methodName, ResultType.AVERAGE_AUC_EVALUATION_DATA_SET.name().toLowerCase(), null, experimentIndex, largestVectorSize) + "\n");
                        averageAreaUnderCurveYLimitMax = Math.max(averageAreaUnderCurveYLimitMax, result.averageEvaluationAreasUnderTheCurve.values().stream().mapToDouble(x -> x).max().orElse(0.0));
                        averageAreaUnderCurveYLimitMin = Math.min(averageAreaUnderCurveYLimitMin, result.averageEvaluationAreasUnderTheCurve.values().stream().mapToDouble(x -> x).min().orElse(0.0));
                        xLimit = Math.max(xLimit, result.averageEvaluationAreasUnderTheCurve.keySet().stream().mapToInt(x -> x).max().orElse(0));
                    }
                    if (resultTypes.contains(ResultType.PREDICTION_ACCURACY_FULL_DATA_SET)) {
                        writer.write(simpleMapToMatlabString(result.predictionAccuracies, methodName, ResultType.PREDICTION_ACCURACY_FULL_DATA_SET.name().toLowerCase(), "1.0", experimentIndex, largestVectorSize) + "\n");
                        predictionAccuracyYLimitMax = Math.max(predictionAccuracyYLimitMax, result.predictionAccuracies.values().stream().mapToDouble(x -> x).max().orElse(0.0));
                        predictionAccuracyYLimitMin = Math.min(predictionAccuracyYLimitMin, result.predictionAccuracies.values().stream().mapToDouble(x -> x).min().orElse(0.0));
                        xLimit = Math.max(xLimit, result.predictionAccuracies.keySet().stream().mapToInt(x -> x).max().orElse(0));
                    }
                    if (resultTypes.contains(ResultType.PREDICTION_ACCURACY_TESTING_DATA_SET)) {
                        writer.write(simpleMapToMatlabString(result.testingPredictionAccuracies, methodName, ResultType.PREDICTION_ACCURACY_TESTING_DATA_SET.name().toLowerCase(), "1.0", experimentIndex, largestVectorSize) + "\n");
                        predictionAccuracyYLimitMax = Math.max(predictionAccuracyYLimitMax, result.testingPredictionAccuracies.values().stream().mapToDouble(x -> x).max().orElse(0.0));
                        predictionAccuracyYLimitMin = Math.min(predictionAccuracyYLimitMin, result.testingPredictionAccuracies.values().stream().mapToDouble(x -> x).min().orElse(0.0));
                        xLimit = Math.max(xLimit, result.testingPredictionAccuracies.keySet().stream().mapToInt(x -> x).max().orElse(0));
                    }
                    if (resultTypes.contains(ResultType.PREDICTION_ACCURACY_EVALUATION_DATA_SET)) {
                        writer.write(simpleMapToMatlabString(result.evaluationPredictionAccuracies, methodName, ResultType.PREDICTION_ACCURACY_EVALUATION_DATA_SET.name().toLowerCase(), null, experimentIndex, largestVectorSize) + "\n");
                        predictionAccuracyYLimitMax = Math.max(predictionAccuracyYLimitMax, result.evaluationPredictionAccuracies.values().stream().mapToDouble(x -> x).max().orElse(0.0));
                        predictionAccuracyYLimitMin = Math.min(predictionAccuracyYLimitMin, result.evaluationPredictionAccuracies.values().stream().mapToDouble(x -> x).min().orElse(0.0));
                        xLimit = Math.max(xLimit, result.evaluationPredictionAccuracies.keySet().stream().mapToInt(x -> x).max().orElse(0));
                    }
                    if (resultTypes.contains(ResultType.NUMBER_OF_EXAMPLES_PICKED)) {
                        writer.write(simpleMapToMatlabString(result.numberOfExamplesPicked, methodName, ResultType.NUMBER_OF_EXAMPLES_PICKED.name().toLowerCase(), null, experimentIndex, largestVectorSize) + "\n");
                        numberOfFixedExamplesYLimitMax = Math.max(numberOfFixedExamplesYLimitMax, result.numberOfExamplesPicked.values().stream().mapToDouble(x -> x).max().orElse(0.0));
                    }
                    if (resultTypes.contains(ResultType.ACTIVE_LEARNING_METHOD_TIMES))
                        writer.write(simpleMapToMatlabString(result.activeLearningMethodTimesTaken, methodName, ResultType.ACTIVE_LEARNING_METHOD_TIMES.name().toLowerCase(), "0.0", experimentIndex, largestVectorSize) + "\n");
                    if (resultTypes.contains(ResultType.TOTAL_TIME_TAKEN))
                        writer.write("times_" + methodName + "(" + experimentIndex + ") = " + (int) Math.floor(result.timeTaken / 1000) + ";\n");
                    experimentIndex++;
                }
            }
            writer.write("\n% Plot results\n");
            writer.write("figure;\n");
            writer.write("setappdata(gcf, 'SubplotDefaultAxesLocation', [0.05, 0.3, 0.9, 0.7]);\n");
            int plotIndex = 1;
            int pDefined = -1;
            double auc1YLimitMax = 0;
            for (ResultType resultType : resultTypes) {
                if (resultType != ResultType.TOTAL_TIME_TAKEN) {
                    writer.write("subplot(1, " + resultTypes.size() + ", " + plotIndex + ");\n"); // TODO: Fix the size() bug.
                    writer.write("hold on;\n");
                    if (resultType != ResultType.AUC_1_ITERATIONS_FULL_DATA_SET
                            && resultType != ResultType.AUC_1_ITERATIONS_TESTING_DATA_SET
                            && resultType != ResultType.AUC_1_ITERATIONS_EVALUATION_DATA_SET) {
                        for (Map.Entry<ScoringFunction, List<ExperimentResults>> resultsEntry : results.entrySet()) {
                            String methodName = resultsEntry.getKey().toString().toLowerCase().replace("-", "_");
                            if (pDefined < 0 || pDefined == plotIndex) {
                                writer.write("p(" + scoringFunctionIndexMap.get(resultsEntry.getKey()) + ") = plot(x, " +
                                        "mean(y_" + methodName + "_" + resultType.name().toLowerCase() +
                                        ", 1), 'Color', [" + matlabPlotColorsMap.get(resultsEntry.getKey()) +
                                        ", 0.8], 'LineWidth', 3);\n");
                                pDefined = plotIndex;
                            } else {
                                writer.write("plot(x, mean(y_" + methodName + "_" + resultType.name().toLowerCase() +
                                        ", 1), 'Color', [" + matlabPlotColorsMap.get(resultsEntry.getKey()) +
                                        ", 0.8], 'LineWidth', 3);\n");
                            }
                        }
                    } else {
                        StringJoiner xTicks = new StringJoiner(",", "[", "]");
                        StringJoiner xValues = new StringJoiner(",", "{", "}");
                        StringJoiner yValues = new StringJoiner(",", "[", "]");
                        StringBuilder plotColors = new StringBuilder();
                        int currentMethod = 1;
                        for (Map.Entry<ScoringFunction, List<ExperimentResults>> resultsEntry : results.entrySet()) {
                            xTicks.add(String.valueOf(currentMethod));
                            xValues.add("'" + resultsEntry.getKey().toString() + "'");
                            plotColors.append("set(h(")
                                    .append(currentMethod++)
                                    .append("), 'FaceColor', [")
                                    .append(matlabPlotColorsMap.get(resultsEntry.getKey()))
                                    .append("]);\n");
                            auc1YLimitMax = Math.max(auc1YLimitMax,
                                    resultsEntry.getValue()
                                            .stream()
                                            .mapToDouble(result -> result.auc1IterationsFull)
                                            .max()
                                            .orElse(0.0));
                            auc1YLimitMax = Math.max(auc1YLimitMax,
                                    resultsEntry.getValue()
                                            .stream()
                                            .mapToDouble(result -> result.auc1IterationsTesting)
                                            .max()
                                            .orElse(0.0));
                            auc1YLimitMax = Math.max(auc1YLimitMax,
                                    resultsEntry.getValue()
                                            .stream()
                                            .mapToDouble(result -> result.auc1IterationsEvaluation)
                                            .max()
                                            .orElse(0.0));
                            switch (resultType) {
                                case AUC_1_ITERATIONS_FULL_DATA_SET:
                                    yValues.add(String.valueOf(resultsEntry.getValue()
                                            .stream()
                                            .mapToDouble(result -> result.auc1IterationsFull)
                                            .average()
                                                                       .orElse(0.0)));
                                    break;
                                case AUC_1_ITERATIONS_TESTING_DATA_SET:
                                    yValues.add(String.valueOf(resultsEntry.getValue()
                                                                       .stream()
                                                                       .mapToDouble(result -> result.auc1IterationsTesting)
                                                                       .average()
                                                                       .orElse(0.0)));
                                    break;
                                case AUC_1_ITERATIONS_EVALUATION_DATA_SET:
                                    yValues.add(String.valueOf(resultsEntry.getValue()
                                                                       .stream()
                                                                       .mapToDouble(result -> result.auc1IterationsEvaluation)
                                                                       .average()
                                                                       .orElse(0.0)));
                                    break;
                            }
                        }
                        writer.write("h = bar(" + xTicks.toString() + ", diag(" + yValues.toString() + "), 'Stacked', 'EdgeColor', 'None', 'BarWidth', 1);\n");
//                        writer.write("set(gca, 'XTick', " + xTicks.toString() + ", 'XTickLabel', " + xValues.toString() + ", 'XTickLabelRotation', -45);\n");
                        writer.write("set(gca, 'XTick', []);\n");
                        writer.write(plotColors.toString());
                    }
                    if (includeTitle)
                        switch (resultType) {
                            case AVERAGE_AUC_FULL_DATA_SET:
                                writer.write("title({'Average AUC Full', ''});\n");
                                break;
                            case AVERAGE_AUC_TESTING_DATA_SET:
                                writer.write("title({'Average AUC Testing', ''});\n");
                                break;
                            case AVERAGE_AUC_EVALUATION_DATA_SET:
                                writer.write("title({'Average AUC Evaluation', ''});\n");
                                break;
                            case AUC_1_ITERATIONS_FULL_DATA_SET:
                                writer.write("title({'Iterations until Average AUC Full=1', ''});\n");
                                break;
                            case AUC_1_ITERATIONS_TESTING_DATA_SET:
                                writer.write("title({'Iterations until Average AUC Testing=1', ''});\n");
                                break;
                            case AUC_1_ITERATIONS_EVALUATION_DATA_SET:
                                writer.write("title({'Iterations until Average AUC Evaluation=1', ''});\n");
                                break;
                            case PREDICTION_ACCURACY_FULL_DATA_SET:
                                writer.write("title({'Prediction Accuracy Full', ''});\n");
                                break;
                            case PREDICTION_ACCURACY_TESTING_DATA_SET:
                                writer.write("title({'Prediction Accuracy Testing', ''});\n");
                                break;
                            case PREDICTION_ACCURACY_EVALUATION_DATA_SET:
                                writer.write("title({'Prediction Accuracy Evaluation', ''});\n");
                                break;
                            case NUMBER_OF_EXAMPLES_PICKED:
                                writer.write("title({'Number of Fixed Labels', ''});\n");
                                break;
                            case ACTIVE_LEARNING_METHOD_TIMES:
                                writer.write("title({'Time Spent in Active Learning Method', ''});\n");
                                break;
                        }
                    if (includeHorizontalAxisLabel)
                        if (resultType != ResultType.AUC_1_ITERATIONS_FULL_DATA_SET
                                && resultType != ResultType.AUC_1_ITERATIONS_TESTING_DATA_SET
                                && resultType != ResultType.AUC_1_ITERATIONS_EVALUATION_DATA_SET)
                            writer.write("xlabel('Iteration Number', 'FontSize', 22);\n");
//                        else
//                            writer.write("xlabel('Method', 'FontSize', 22);\n");
                    if (includeVerticalAxisLabel)
                        switch (resultType) {
                            case AVERAGE_AUC_FULL_DATA_SET:
                            case AVERAGE_AUC_TESTING_DATA_SET:
                            case AVERAGE_AUC_EVALUATION_DATA_SET:
//                                writer.write("ylabel('Average AUC', 'FontSize', 22);\n");
                                writer.write("xlim([0 " + (2 * xLimit) + "]);\n");
                                writer.write("ylim([" + averageAreaUnderCurveYLimitMin + " " + averageAreaUnderCurveYLimitMax + "]);\n");
                                writer.write("set(gca, 'XTick', round(linspace(0, " + (2 * xLimit) + ", 3)));\n");
                                writer.write("set(gca, 'YTick', round(linspace(" + averageAreaUnderCurveYLimitMin + ", " + averageAreaUnderCurveYLimitMax + ", 3), 2));\n");
                                break;
                            case AUC_1_ITERATIONS_FULL_DATA_SET:
                            case AUC_1_ITERATIONS_TESTING_DATA_SET:
                            case AUC_1_ITERATIONS_EVALUATION_DATA_SET:
//                                writer.write("ylabel('Number of Iterations');\n");
                                writer.write("ylim([0 " + auc1YLimitMax + "]);\n");
                                writer.write("set(gca, 'YTick', round(linspace(0, " + auc1YLimitMax + ", 3)));\n");
                                break;
                            case PREDICTION_ACCURACY_FULL_DATA_SET:
                            case PREDICTION_ACCURACY_TESTING_DATA_SET:
                            case PREDICTION_ACCURACY_EVALUATION_DATA_SET:
//                                writer.write("ylabel('Prediction Accuracy', 'FontSize', 22);\n");
                                writer.write("xlim([0 " + (2 * xLimit) + "]);\n");
                                writer.write("ylim([" + predictionAccuracyYLimitMin + " " + predictionAccuracyYLimitMax + "]);\n");
                                writer.write("set(gca, 'XTick', round(linspace(0, " + (2 * xLimit) + ", 3)));\n");
                                writer.write("set(gca, 'YTick', round(linspace(" + predictionAccuracyYLimitMin + ", " + predictionAccuracyYLimitMax + ", 3), 2));\n");
                                break;
                            case NUMBER_OF_EXAMPLES_PICKED:
//                                writer.write("ylabel('Number of Fixed Labels', 'FontSize', 22);\n");
                                writer.write("xlim([0 " + (2 * xLimit) + "]);\n");
                                writer.write("ylim([0 " + numberOfFixedExamplesYLimitMax + "]);\n");
                                writer.write("set(gca, 'XTick', round(linspace(0, " + (2 * xLimit) + ", 3)));\n");
                                writer.write("set(gca, 'YTick', round(linspace(0, " + numberOfFixedExamplesYLimitMax + ", 3)));\n");
                                break;
                            case ACTIVE_LEARNING_METHOD_TIMES:
//                                writer.write("ylabel('Time Spent in Active Learning Method', 'FontSize', 22);\n");
                                writer.write("xlim([0 " + (2 * xLimit) + "]);\n");
                                writer.write("set(gca, 'XTick', round(linspace(0, " + (2 * xLimit) + ", 3)));\n");
                                break;
                        }
                    writer.write("set(gca, 'FontSize', 22);\n");
                    writer.write("hold off;\n");
                    plotIndex++;
                }
            }
            if (includeLegend) {
                StringJoiner legendPlotNames = new StringJoiner(", ", "[", "]");
                StringJoiner legendPlotDescriptions = new StringJoiner(", ", "{", "}");
                if (resultTypes.contains(ResultType.AUC_1_ITERATIONS_FULL_DATA_SET)
                        || resultTypes.contains(ResultType.AUC_1_ITERATIONS_FULL_DATA_SET)
                        || resultTypes.contains(ResultType.AUC_1_ITERATIONS_FULL_DATA_SET)) {
                    for (ScoringFunction scoringFunction : results.keySet()) {
//                        String methodName = scoringFunction.toString().toLowerCase().replace("-", "_");
//                        legendPlotDescriptions.add("strcat(['" + scoringFunction.toString() + " (' " +
//                                                           "num2str(trapz(x, mean(y_" + methodName + "_" +
//                                                           ResultType.AVERAGE_AUC_FULL_DATA_SET.name().toLowerCase() + // TODO: Fix small bug that occurs if average AUC over the full data set is not included in the results.
//                                                           ", 1)), '%1.3f') ')'])");
                        legendPlotNames.add("h(" + scoringFunctionIndexMap.get(scoringFunction) + ")");
                        legendPlotDescriptions.add("'" + scoringFunction.toString() + "'");
                    }
                } else {
                    for (ScoringFunction scoringFunction : results.keySet()) {
//                        String methodName = scoringFunction.toString().toLowerCase().replace("-", "_");
//                        legendPlotDescriptions.add("strcat(['" + scoringFunction.toString() + " (' " +
//                                                           "num2str(trapz(x, mean(y_" + methodName + "_" +
//                                                           ResultType.AVERAGE_AUC_FULL_DATA_SET.name().toLowerCase() + // TODO: Fix small bug that occurs if average AUC over the full data set is not included in the results.
//                                                           ", 1)), '%1.3f') ')'])");
                        legendPlotNames.add("p(" + scoringFunctionIndexMap.get(scoringFunction) + ")");
                        legendPlotDescriptions.add("'" + scoringFunction.toString() + "'");
                    }
                }
                writer.write("legend(" + legendPlotNames.toString() + ", " + legendPlotDescriptions.toString() + ", "
                                     + "'Position', [0.25 0.025 0.50 0.025]', 'Orientation', 'Horizontal', 'Box', 'Off');\n");
            }
            writer.write("set(gcf, 'Position', [0, 0, 1500, 280], 'PaperUnits', 'points', 'PaperPosition', [0, 0, 1500, 280], 'PaperSize', [1500, 280]);\n" +
                                 "saveas(gcf, '" + filePath.substring(filePath.lastIndexOf("/") + 1, filePath.length() - 2) + ".pdf');");
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
        private final Map<DataInstance<Vector>, Map<Label, Boolean>> trainingDataSet;
        private final Map<DataInstance<Vector>, Map<Label, Boolean>> testingDataSet;
        private final Map<Label, DataSetStatistics> trainingDataSetStatistics;
        private final Map<Label, DataSetStatistics> testingDataSetStatistics;
        private final double l1RegularizationWeight;
        private final double l2RegularizationWeight;

        public ImportedDataSet(Set<Label> labels,
                               ConstraintSet constraints,
                               Map<DataInstance<Vector>, Map<Label, Boolean>> trainingDataSet,
                               Map<DataInstance<Vector>, Map<Label, Boolean>> testingDataSet) {
            this(labels, constraints, trainingDataSet, testingDataSet, 0.0, 0.0);
        }

        public ImportedDataSet(Set<Label> labels,
                               ConstraintSet constraints,
                               Map<DataInstance<Vector>, Map<Label, Boolean>> trainingDataSet,
                               Map<DataInstance<Vector>, Map<Label, Boolean>> testingDataSet,
                               double l1RegularizationWeight,
                               double l2RegularizationWeight) {
            this.labels = labels;
            this.constraints = constraints;
            this.trainingDataSet = trainingDataSet;
            this.testingDataSet = testingDataSet;
            trainingDataSetStatistics = new HashMap<>();
            for (Label label : labels)
                trainingDataSetStatistics.put(label, new DataSetStatistics());
            for (Map.Entry<DataInstance<Vector>, Map<Label, Boolean>> instanceEntry : trainingDataSet.entrySet()) {
                for (Map.Entry<Label, Boolean> instanceLabelEntry : instanceEntry.getValue().entrySet()) {
                    if (instanceLabelEntry.getValue())
                        trainingDataSetStatistics.get(instanceLabelEntry.getKey()).numberOfPositiveExamples++;
                    else
                        trainingDataSetStatistics.get(instanceLabelEntry.getKey()).numberOfNegativeExamples++;
                    trainingDataSetStatistics.get(instanceLabelEntry.getKey()).totalNumberOfExamples++;
                }
            }
            testingDataSetStatistics = new HashMap<>();
            for (Label label : labels)
                testingDataSetStatistics.put(label, new DataSetStatistics());
            for (Map.Entry<DataInstance<Vector>, Map<Label, Boolean>> instanceEntry : testingDataSet.entrySet()) {
                for (Map.Entry<Label, Boolean> instanceLabelEntry : instanceEntry.getValue().entrySet()) {
                    if (instanceLabelEntry.getValue())
                        testingDataSetStatistics.get(instanceLabelEntry.getKey()).numberOfPositiveExamples++;
                    else
                        testingDataSetStatistics.get(instanceLabelEntry.getKey()).numberOfNegativeExamples++;
                    testingDataSetStatistics.get(instanceLabelEntry.getKey()).totalNumberOfExamples++;
                }
            }
            this.l1RegularizationWeight = l1RegularizationWeight;
            this.l2RegularizationWeight = l2RegularizationWeight;
        }

        public Set<Label> getLabels() {
            return labels;
        }

        public Map<DataInstance<Vector>, Map<Label, Boolean>> getTrainingDataSet() {
            return trainingDataSet;
        }

        public Map<DataInstance<Vector>, Map<Label, Boolean>> getTestingDataSet() {
            return testingDataSet;
        }

        public Map<Label, DataSetStatistics> getStatistics() {
            return trainingDataSetStatistics;
        }

        public void exportStatistics(String filePath) {
            try {
                FileWriter writer = new FileWriter(filePath);
                writer.write("Training Data Set:\n");
                for (Label label : labels)
                    writer.write("\t" + label.name() + ": {" + trainingDataSetStatistics.get(label).toString() + " }\n");
                writer.write("Testing Data Set:\n");
                for (Label label : labels)
                    writer.write("\t" + label.name() + ": {" + testingDataSetStatistics.get(label).toString() + " }\n");
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
        AUC_1_ITERATIONS_FULL_DATA_SET,
        AUC_1_ITERATIONS_TESTING_DATA_SET,
        AUC_1_ITERATIONS_EVALUATION_DATA_SET,
        PREDICTION_ACCURACY_FULL_DATA_SET,
        PREDICTION_ACCURACY_TESTING_DATA_SET,
        PREDICTION_ACCURACY_EVALUATION_DATA_SET,
        NUMBER_OF_EXAMPLES_PICKED,
        ACTIVE_LEARNING_METHOD_TIMES,
        TOTAL_TIME_TAKEN
    }

    private static class ExperimentResults {
        private Map<Integer, Double> averageAreasUnderTheCurve = new HashMap<>();
        private Map<Integer, Double> averageTestingAreasUnderTheCurve = new HashMap<>();
        private Map<Integer, Double> averageEvaluationAreasUnderTheCurve = new HashMap<>();
        private int auc1IterationsFull;
        private int auc1IterationsTesting;
        private int auc1IterationsEvaluation;
        private Map<Integer, Double> predictionAccuracies = new HashMap<>();
        private Map<Integer, Double> testingPredictionAccuracies = new HashMap<>();
        private Map<Integer, Double> evaluationPredictionAccuracies = new HashMap<>();
        private Map<Integer, Integer> numberOfExamplesPicked = new HashMap<>();
        private Map<Integer, Long> activeLearningMethodTimesTaken = new HashMap<>();
        private long timeTaken;
    }

    private static ImportedDataSet importISOLETDataSet(String workingDirectory,
                                                       double negativeToPositiveTrainingExamplesRatio,
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
        Map<DataInstance<Vector>, Map<Label, Boolean>> trainingDataSet = new ConcurrentHashMap<>();
        Set<Label> labels = labeledInstances.values().stream().map(Label::new).collect(Collectors.toSet());
        Map<Label, Integer> numberOfPositiveExamples = new HashMap<>();
        Map<Label, List<DataInstance<Vector>>> negativeTrainingExamples = new ConcurrentHashMap<>();
        for (Label label : labels) {
            numberOfPositiveExamples.put(label, 0);
            negativeTrainingExamples.put(label, new ArrayList<>());
        }
        Set<String> uniqueNames = new HashSet<>();
        for (Map.Entry<Vector, String> labeledInstanceEntry : labeledInstances.entrySet()) {
            Vector features = labeledInstanceEntry.getKey();
            String name = labeledInstanceEntry.getValue() + ":" + features.toString();
            DataInstance<Vector> dataInstance = new DataInstance<>(name, features);
            if (!uniqueNames.contains(name)) {
                uniqueNames.add(name);
                String labelName = labeledInstanceEntry.getValue();
                Label label = new Label(labelName);
                Set<String> negativeLabels = labels.stream().map(Label::name).collect(Collectors.toSet());
                negativeLabels.remove(labelName);
                if (!trainingDataSet.containsKey(dataInstance))
                    trainingDataSet.put(dataInstance, new HashMap<>());
                trainingDataSet.get(dataInstance).put(label, true);
                numberOfPositiveExamples.put(label, numberOfPositiveExamples.get(label) + 1);
                for (String negativeLabelName : negativeLabels)
                    negativeTrainingExamples.get(new Label(negativeLabelName)).add(dataInstance);
            }
        }
        for (Label label : labels) {
            int numberOfNegativeExamples = (int) negativeToPositiveTrainingExamplesRatio * numberOfPositiveExamples.get(label);
            Collections.shuffle(negativeTrainingExamples.get(label));
            for (int index = 0; index < Math.min(numberOfNegativeExamples, negativeTrainingExamples.get(label).size()); index++)
                trainingDataSet.get(negativeTrainingExamples.get(label).get(index)).put(label, false);
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
                        Set<String> negativeLabels = labels.stream().map(Label::name).collect(Collectors.toSet());
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
                                   trainingDataSet,
                                   evaluationDataSet,
                                   l1RegularizationWeight,
                                   l2RegularizationWeight);
    }

    private static ImportedDataSet importLIBSVMDataSet(String workingDirectory,
                                                       boolean sparseFeatures,
                                                       double negativeToPositiveTrainingExamplesRatio,
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
        Map<DataInstance<Vector>, Map<Label, Boolean>> trainingDataSet = new ConcurrentHashMap<>();
        Set<Label> labels = labeledInstances.values().stream().map(Label::new).collect(Collectors.toSet());
        Map<Label, Integer> numberOfPositiveExamples = new HashMap<>();
        Map<Label, List<DataInstance<Vector>>> negativeTrainingExamples = new ConcurrentHashMap<>();
        for (Label label : labels) {
            numberOfPositiveExamples.put(label, 0);
            negativeTrainingExamples.put(label, new ArrayList<>());
        }
        Set<String> uniqueNames = new HashSet<>();
        for (Map.Entry<Vector, String> labeledInstanceEntry : labeledInstances.entrySet()) {
            Vector features = labeledInstanceEntry.getKey();
            String name = labeledInstanceEntry.getValue() + ":" + features.toString();
            DataInstance<Vector> dataInstance = new DataInstance<>(name, features);
            if (!uniqueNames.contains(name)) {
                uniqueNames.add(name);
                String labelName = labeledInstanceEntry.getValue();
                Label label = new Label(labelName);
                Set<String> negativeLabels = labels.stream().map(Label::name).collect(Collectors.toSet());
                negativeLabels.remove(labelName);
                if (!trainingDataSet.containsKey(dataInstance))
                    trainingDataSet.put(dataInstance, new HashMap<>());
                trainingDataSet.get(dataInstance).put(label, true);
                numberOfPositiveExamples.put(label, numberOfPositiveExamples.get(label) + 1);
                for (String negativeLabelName : negativeLabels)
                    negativeTrainingExamples.get(new Label(negativeLabelName)).add(dataInstance);
            }
        }
        for (Label label : labels) {
            int numberOfNegativeExamples = (int) negativeToPositiveTrainingExamplesRatio * numberOfPositiveExamples.get(label);
            Collections.shuffle(negativeTrainingExamples.get(label));
            for (int index = 0; index < Math.min(numberOfNegativeExamples, negativeTrainingExamples.get(label).size()); index++)
                trainingDataSet.get(negativeTrainingExamples.get(label).get(index)).put(label, false);
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
                        Set<String> negativeLabels = labels.stream().map(Label::name).collect(Collectors.toSet());
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
                                   trainingDataSet,
                                   evaluationDataSet,
                                   l1RegularizationWeight,
                                   l2RegularizationWeight);
    }

    private static ImportedDataSet importNELLDataSet(String cplFeatureMapDirectory,
                                                     String workingDirectory,
                                                     double negativeToPositiveTrainingExamplesRatio,
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
        Map<DataInstance<Vector>, Map<Label, Boolean>> trainingDataSet = new ConcurrentHashMap<>();
        Map<DataInstance<Vector>, Map<Label, Boolean>> evaluationDataSet = new ConcurrentHashMap<>();
        Set<String> nounPhrasesWithoutFeatures = new HashSet<>();
        Map<String, Set<String>> filteredLabeledNounPhrases = new ConcurrentHashMap<>();
        Map<String, Set<String>> filteredEvaluationNounPhrases = new ConcurrentHashMap<>();
        Map<Label, Integer> numberOfPositiveExamples = new HashMap<>();
        Map<Label, List<DataInstance<Vector>>> negativeTrainingExamples = new ConcurrentHashMap<>();
        for (Map.Entry<String, Set<String>> labeledNounPhraseEntry : labeledNounPhrases.entrySet()) {
            String[] nounPhraseParts = labeledNounPhraseEntry.getKey().split("\\|");
            String nounPhraseType = nounPhraseParts[0];
            String nounPhrase = nounPhraseParts[1];
            Set<String> positiveLabels = labeledNounPhraseEntry.getValue();
            Set<String> negativeLabels = labels.stream().map(Label::name).collect(Collectors.toSet());
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
                if (!trainingDataSet.containsKey(dataInstance))
                    trainingDataSet.put(dataInstance, new HashMap<>());
                positiveLabels.forEach(labelName -> {
                    Label label = new Label(labelName);
                    trainingDataSet.get(dataInstance).put(label, true);
                    if (!numberOfPositiveExamples.containsKey(label))
                        numberOfPositiveExamples.put(label, 1);
                    else
                        numberOfPositiveExamples.put(label, numberOfPositiveExamples.get(label) + 1);
                });
                negativeLabels.forEach(labelName -> {
                    Label label = new Label(labelName);
                    if (!negativeTrainingExamples.containsKey(label))
                        negativeTrainingExamples.put(label, new ArrayList<>());
                    negativeTrainingExamples.get(label).add(dataInstance);
                });
            } else if (nounPhraseType.equals("EVALUATION")) {
                if (!evaluationDataSet.containsKey(dataInstance))
                    evaluationDataSet.put(dataInstance, new HashMap<>());
                for (String labelName : positiveLabels)
                    evaluationDataSet.get(dataInstance).put(new Label(labelName), true);
                for (String labelName : negativeLabels)
                    evaluationDataSet.get(dataInstance).put(new Label(labelName), false);
            }
        }
        for (Label label : labels) {
            int numberOfNegativeExamples = (int) negativeToPositiveTrainingExamplesRatio * numberOfPositiveExamples.get(label);
            Collections.shuffle(negativeTrainingExamples.get(label));
            for (int index = 0; index < Math.min(numberOfNegativeExamples, negativeTrainingExamples.get(label).size()); index++)
                trainingDataSet.get(negativeTrainingExamples.get(label).get(index)).put(label, false);
        }
//        if (nounPhrasesWithoutFeatures.size() > 0)
//            logger.info("NELL noun phrases without features that were ignored: " + nounPhrasesWithoutFeatures);
//        else
//            logger.info("There were no NELL noun phrases without features in the provided data.");
        exportLabeledNounPhrases(filteredLabeledNounPhrases, workingDirectory + "/filtered_labeled_nps.tsv");
        exportLabeledNounPhrases(filteredEvaluationNounPhrases, workingDirectory + "/filtered_evaluation_nps.tsv");
        return new ImportedDataSet(labels,
                                   importConstraints(workingDirectory),
                                   trainingDataSet,
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
//            Stream<String> contextsLines = new BufferedReader(new InputStreamReader(new GZIPInputStream(
//                    Files.newInputStream(Paths.get(cplFeatureMapDirectory + "/cat_contexts.txt.gz"))
//            ))).lines();
//            Map<String, Integer> contextCounts = new HashMap<>();
//            contextsLines.forEach(line -> {
//                String[] lineParts = line.split("\t");
//                contextCounts.put(lineParts[0], Integer.parseInt(lineParts[1]));
//            });
//            List<Integer> contextCountsList = new ArrayList<>();
//            contextCounts.entrySet()
//                    .stream()
//                    .sorted(Collections.reverseOrder(Comparator.comparing(Map.Entry::getValue)))
//                    .forEachOrdered(e -> contextCountsList.add(e.getValue()));
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
//                        if (contextCounts.get(contextParts[0]) > contextCountsList.get(0) * 0.01)
                        contextValues.put(contextParts[0], Double.parseDouble(contextParts[1]));
                    }
                    if (contextValues.size() > 0)
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
            ScoringFunction[] scoringFunctions,
            ExamplePickingMethod examplePickingMethod,
            boolean useLogicIntegrator,
            boolean retrainClassifiers,
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
                ConstrainedLearningExperiment experiment = new ConstrainedLearningExperiment(
                        numberOfExamplesToPickPerIteration,
                        maximumNumberOfIterations,
                        scoringFunction,
                        examplePickingMethod,
                        useLogicIntegrator,
                        retrainClassifiers,
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
        Map<ScoringFunction, List<ExperimentResults>> sortedResults = new LinkedHashMap<>();
        for (ScoringFunction scoringFunction : scoringFunctions)
            sortedResults.put(scoringFunction, results.get(scoringFunction));
        logger.info("Finished!");
        return sortedResults;
    }

    public static String execReadToString(String execCommand) {
        try {
            Process proc = Runtime.getRuntime().exec(execCommand);
            try (InputStream stream = proc.getInputStream()) {
                try (Scanner s = new Scanner(stream).useDelimiter("\n")) {
                    return s.hasNext() ? s.next() : "";
                }
            } catch (Exception e) {
                logger.error(e);
            }
        } catch (Exception e) {
            logger.error(e);
        }
        return null;
    }

    public static void main(String[] args) {
        int numberOfExperimentRepetitions = 2;
        int numberOfExamplesToPickPerIteration = 1000000;
        int maximumNumberOfIterations = 1000000000;
        boolean includeTitleInResultsPlot = true;
        boolean includeHorizontalAxisLabelInResultsPlot = true;
        boolean includeVerticalAxisLabelInResultsPlot = true;
        boolean includeLegendInResultsPlot = true;
        ScoringFunction[] scoringFunctions = new ScoringFunction[] {
                new RandomScoringFunction(),
                new EntropyScoringFunction(),
                new RandomScoringFunction(true),
                new EntropyScoringFunction(true),
                new ExpectedLabelsFixedScoringFunction(true),
                new ConstraintPropagationScoringFunction(true),
//                new ConstraintPropagationScoringFunction(SurpriseFunction.NEGATIVE_LOGARITHM, false),
//                new ConstraintPropagationScoringFunction(SurpriseFunction.ONE_MINUS_PROBABILITY, false),
                new ConstraintPropagationScoringFunction(SurpriseFunction.NEGATIVE_LOGARITHM, true),
                new ConstraintPropagationScoringFunction(SurpriseFunction.ONE_MINUS_PROBABILITY, true)
        };
        Set<ResultType> resultTypes = new LinkedHashSet<>();
//        resultTypes.add(ResultType.AUC_1_ITERATIONS_FULL_DATA_SET);
        resultTypes.add(ResultType.AVERAGE_AUC_FULL_DATA_SET);
//        resultTypes.add(ResultType.AVERAGE_AUC_FULL_DATA_SET);
        resultTypes.add(ResultType.AVERAGE_AUC_EVALUATION_DATA_SET);
//        resultTypes.add(ResultType.PREDICTION_ACCURACY_FULL_DATA_SET);
        resultTypes.add(ResultType.NUMBER_OF_EXAMPLES_PICKED);
        boolean useLogicIntegrator = args[0].equals("1");
        boolean retrainClassifiers = args[1].equals("1");
        double negativeToPositiveTrainingExamplesRatio = 1;
        ExamplePickingMethod examplePickingMethod = ExamplePickingMethod.PSEUDO_SEQUENTIAL;

        String experiment = args[2].replace("'", "");
        String workingDirectory;
        String cplFeatureMapDirectory;
        String hostname = execReadToString("hostname");
        switch (hostname) {
            case "Emmanouils-MBP.wv.cc.cmu.edu":
                workingDirectory = "/Users/Anthony/Development/Data/NELL/Active Learning Experiment/";
                cplFeatureMapDirectory = "/Users/Anthony/Development/Data/NELL/Server/all-pairs/all-pairs-OC-2011-02-02-smallcontexts50-gz";
                break;
            case "gpu1.learning.cs.cmu.edu":
            case "gpu2.learning.cs.cmu.edu":
            case "gpu3.learning.cs.cmu.edu":
                workingDirectory = "/home/eplatani/data/active_learning/";
                cplFeatureMapDirectory = "/home/eplatani/data/nell/all_pairs/all-pairs-OC-2011-02-02-smallcontexts50-gz";
                break;
            default:
                throw new IllegalStateException("Unsupported hostname \"" + hostname + "\".");
        }
        ImportedDataSet dataSet;
        Map<ScoringFunction, List<ExperimentResults>> results;

        // NELL Data Set Experiment
        logger.info("Running " + experiment + " experiment...");
        workingDirectory += "Experiment " + experiment + " - LogicIteration";
        numberOfExperimentRepetitions = 10;
        switch (experiment) {
            case "NELL 7 Class":
            case "NELL 11 Class":
            case "NELL 13 Class":
                numberOfExamplesToPickPerIteration = 1000;
                maximumNumberOfIterations = 1000;
                dataSet = importNELLDataSet(cplFeatureMapDirectory, workingDirectory, negativeToPositiveTrainingExamplesRatio, 0, 0);
                break;
            case "IRIS":
                numberOfExamplesToPickPerIteration = 1;
                maximumNumberOfIterations = 1000;
                dataSet = importISOLETDataSet(workingDirectory, negativeToPositiveTrainingExamplesRatio, 0.0, 0.0);
                break;
            case "DNA":
                numberOfExamplesToPickPerIteration = 50;
                maximumNumberOfIterations = 1000;
                dataSet = importLIBSVMDataSet(workingDirectory, true, negativeToPositiveTrainingExamplesRatio, 0.0, 0.0);
                break;
            case "LETTER":
                numberOfExamplesToPickPerIteration = 1000;
                maximumNumberOfIterations = 100000;
                dataSet = importLIBSVMDataSet(workingDirectory, true, negativeToPositiveTrainingExamplesRatio, 0.0, 0.0);
                break;
            case "PENDIGITS":
                numberOfExamplesToPickPerIteration = 100;
                maximumNumberOfIterations = 100000;
                dataSet = importLIBSVMDataSet(workingDirectory, false, negativeToPositiveTrainingExamplesRatio, 0.0, 0.0);
                break;
            case "ISOLET":
                numberOfExamplesToPickPerIteration = 1000;
                maximumNumberOfIterations = 100000;
                dataSet = importISOLETDataSet(workingDirectory, negativeToPositiveTrainingExamplesRatio, 0.0, 0.0);
                break;
            case "PROTEIN":
                numberOfExamplesToPickPerIteration = 100;
                maximumNumberOfIterations = 2000;
                dataSet = importLIBSVMDataSet(workingDirectory, false, negativeToPositiveTrainingExamplesRatio, 0.0, 0.0);
                break;
            case "SATIMAGE":
                numberOfExamplesToPickPerIteration = 100;
                maximumNumberOfIterations = 100000;
                dataSet = importLIBSVMDataSet(workingDirectory, false, negativeToPositiveTrainingExamplesRatio, 0.0, 0.0);
                break;
            case "VOWEL":
                numberOfExamplesToPickPerIteration = 10;
                maximumNumberOfIterations = 100000;
                dataSet = importLIBSVMDataSet(workingDirectory, false, negativeToPositiveTrainingExamplesRatio, 0.0, 0.0);
                break;
            case "SHUTTLE":
                numberOfExamplesToPickPerIteration = 1000;
                maximumNumberOfIterations = 100000;
                dataSet = importLIBSVMDataSet(workingDirectory, false, negativeToPositiveTrainingExamplesRatio, 0.0, 0.0);
                break;
            case "SENSIT-VEHICLE":
                numberOfExamplesToPickPerIteration = 1000;
                maximumNumberOfIterations = 100000;
                dataSet = importLIBSVMDataSet(workingDirectory, false, negativeToPositiveTrainingExamplesRatio, 0.0, 0.0);
                break;
            case "SENSIT-VEHICLE-SEISMIC":
                numberOfExamplesToPickPerIteration = 1000;
                maximumNumberOfIterations = 10000;
                dataSet = importLIBSVMDataSet(workingDirectory, false, negativeToPositiveTrainingExamplesRatio, 0.0, 0.0);
                break;
            case "WINE":
                numberOfExamplesToPickPerIteration = 1;
                maximumNumberOfIterations = 1000;
                dataSet = importLIBSVMDataSet(workingDirectory, false, negativeToPositiveTrainingExamplesRatio, 0.0, 0.0);
                break;
            case "GLASS":
                numberOfExamplesToPickPerIteration = 10;
                maximumNumberOfIterations = 10000000;
                dataSet = importLIBSVMDataSet(workingDirectory, false, negativeToPositiveTrainingExamplesRatio, 0.0, 0.0);
                break;
            case "VEHICLE":
                numberOfExamplesToPickPerIteration = 10;
                maximumNumberOfIterations = 1000;
                dataSet = importLIBSVMDataSet(workingDirectory, false, negativeToPositiveTrainingExamplesRatio, 0.0, 0.0);
                break;
            case "MNIST":
                numberOfExamplesToPickPerIteration = 10000;
                maximumNumberOfIterations = 100000;
                dataSet = importLIBSVMDataSet(workingDirectory, true, negativeToPositiveTrainingExamplesRatio, 0.0, 0.0);
                break;
            case "SEGMENT":
                numberOfExamplesToPickPerIteration = 100;
                maximumNumberOfIterations = 1000;
                dataSet = importLIBSVMDataSet(workingDirectory, false, negativeToPositiveTrainingExamplesRatio, 0.0, 0.0);
                break;
            case "COVTYPE":
                numberOfExamplesToPickPerIteration = 10000;
                maximumNumberOfIterations = 1000000;
                dataSet = importLIBSVMDataSet(workingDirectory, true, negativeToPositiveTrainingExamplesRatio, 0.0, 0.0);
                break;
            case "NEWS20":
                numberOfExamplesToPickPerIteration = 10000;
                maximumNumberOfIterations = 1000000;
                dataSet = importLIBSVMDataSet(workingDirectory, true, negativeToPositiveTrainingExamplesRatio, 0.0, 0.0);
                break;
            default:
                throw new IllegalStateException("Unsupported dataset name.");
        }

        String resultsDirectoryName;
        if (retrainClassifiers && useLogicIntegrator)
            resultsDirectoryName = "/results_integrator_classifiers_" + examplePickingMethod.name() + ".m";
        else if (retrainClassifiers)
            resultsDirectoryName = "/results_classifiers_" + examplePickingMethod.name() + ".m";
        else if (useLogicIntegrator)
            resultsDirectoryName = "/results_integrator_" + examplePickingMethod.name() + ".m";
        else
            resultsDirectoryName = "/results_plain_" + examplePickingMethod.name() + ".m";

        dataSet.exportStatistics(workingDirectory + "/statistics.txt");
        results = runExperiments(numberOfExperimentRepetitions,
                                 numberOfExamplesToPickPerIteration,
                                 maximumNumberOfIterations,
                                 scoringFunctions,
                                 examplePickingMethod,
                                 useLogicIntegrator,
                                 retrainClassifiers,
                                 dataSet,
                                 resultTypes);
        ConstrainedLearningExperiment.exportResults(
                numberOfExperimentRepetitions,
                numberOfExamplesToPickPerIteration,
                maximumNumberOfIterations,
                results,
                workingDirectory + resultsDirectoryName,
                resultTypes,
                includeTitleInResultsPlot,
                includeHorizontalAxisLabelInResultsPlot,
                includeVerticalAxisLabelInResultsPlot,
                includeLegendInResultsPlot);
        logger.info("Finished all experiments!");
    }
}
