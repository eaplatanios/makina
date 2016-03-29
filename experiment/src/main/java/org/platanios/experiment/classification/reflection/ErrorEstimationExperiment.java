package org.platanios.experiment.classification.reflection;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.experiment.data.DataSets;
import org.platanios.learn.classification.Label;
import org.platanios.learn.classification.constraint.Constraint;
import org.platanios.learn.classification.reflection.*;
import org.platanios.math.StatisticsUtilities;
import org.platanios.utilities.CollectionUtilities;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorEstimationExperiment {
    private static final DecimalFormat DECIMAL_FORMAT = new DecimalFormat("0.0000000000E0");
    private static final DecimalFormat RESULTS_DECIMAL_FORMAT = new DecimalFormat("0.00E0");
    private static final Logger logger = LogManager.getFormatterLogger("Classification / Reflection / Error Estimation Experiment");

    private final Map<Label, Integer> numberOfInstances = new HashMap<>();

    private final Integrator.ErrorRates sampleErrorRates;
    private final Integrator.Data<Integrator.Data.ObservedInstance> observedData;
    private final Integrator.Data<Integrator.Data.PredictedInstance> predictedData;
    private final Set<Label> labels;
    private final Set<Constraint> constraints;
    private final BiMap<String, Integer> instanceIDsMap;
    private final BiMap<String, Integer> componentIDsMap;

    private ErrorEstimationExperiment(Integrator.Data<Integrator.Data.ObservedInstance> observedData,
                                      Integrator.Data<Integrator.Data.PredictedInstance> predictedData,
                                      Set<Constraint> constraints,
                                      BiMap<String, Integer> instanceIDsMap,
                                      BiMap<String, Integer> componentIDsMap) {
        labels = predictedData.stream().map(Integrator.Data.Instance::label).collect(Collectors.toSet());
        sampleErrorRates = computeSampleErrorRates(observedData, predictedData);
        this.observedData = observedData;
        this.predictedData = predictedData;
        this.constraints = constraints;
        this.instanceIDsMap = instanceIDsMap;
        this.componentIDsMap = componentIDsMap;
    }

    private void runExperiment(String methodName, boolean detailedLog) {
        Integrator integrator;
        int experimentRepetitions = 1;
        Matcher patternMatcher = Pattern.compile("([^\\[\\]]*)(?:\\[(.*?)\\])?").matcher(methodName);
        if (patternMatcher.find()) {
            methodName = patternMatcher.group(1);
            if (patternMatcher.group(2) != null)
                experimentRepetitions = Integer.parseInt(patternMatcher.group(2));
        }
        List<Results> results = new ArrayList<>();
        for (int repetition = 0; repetition < experimentRepetitions; repetition++) {
            if (methodName.equals("MVI")) {
                integrator = new MajorityVoteIntegrator.Builder(predictedData).build();
            } else if (methodName.startsWith("AI")) {
                String[] methodNameParts = methodName.split("=");
                if (methodNameParts.length == 1)
                    integrator = new AgreementIntegrator.Builder(predictedData)
                            .highestOrder(-1)
                            .build();
                else
                    integrator = new AgreementIntegrator.Builder(predictedData)
                            .highestOrder(Integer.parseInt(methodNameParts[1]))
                            .build();
            } else if (methodName.startsWith("BI")) {
                String[] methodNameParts = methodName.split("=");
                if (methodNameParts.length == 1)
                    integrator = new BayesianIntegrator.Builder(predictedData).build();
                else
                    integrator = new BayesianIntegrator.Builder(predictedData)
                            .numberOfBurnInSamples(Integer.parseInt(methodNameParts[1]))
                            .numberOfThinningSamples(Integer.parseInt(methodNameParts[2]))
                            .numberOfSamples(Integer.parseInt(methodNameParts[3]))
                            .build();
            } else if (methodName.startsWith("CBI")) {
                String[] methodNameParts = methodName.split("=");
                if (methodNameParts.length == 1)
                    integrator = new CoupledBayesianIntegrator.Builder(predictedData).build();
                else
                    integrator = new CoupledBayesianIntegrator.Builder(predictedData)
                            .numberOfBurnInSamples(Integer.parseInt(methodNameParts[1]))
                            .numberOfThinningSamples(Integer.parseInt(methodNameParts[2]))
                            .numberOfSamples(Integer.parseInt(methodNameParts[3]))
                            .build();
            } else if (methodName.startsWith("HCBI")) {
                String[] methodNameParts = methodName.split("=");
                if (methodNameParts.length == 1)
                    integrator = new HierarchicalCoupledBayesianIntegrator.Builder(predictedData).build();
                else
                    integrator = new HierarchicalCoupledBayesianIntegrator.Builder(predictedData)
                            .numberOfBurnInSamples(Integer.parseInt(methodNameParts[1]))
                            .numberOfThinningSamples(Integer.parseInt(methodNameParts[2]))
                            .numberOfSamples(Integer.parseInt(methodNameParts[3]))
                            .build();
            } else if (methodName.startsWith("LI-TUFFY")) {
                integrator =
                        new TuffyLogicIntegrator.Builder(predictedData)
                                .addConstraints(constraints)
                                .workingDirectory("/Users/Anthony/Desktop/tmp")
                                .logProgress(detailedLog)
                                .build();
            } else if (methodName.startsWith("LI")) {
                String[] methodNameParts = methodName.split("=");
                if (methodNameParts.length == 1) {
                    integrator =
                            new LogicIntegrator.Builder(predictedData)
                                    .addConstraints(constraints)
                                    .logProgress(detailedLog)
                                    .build();
                } else {
                    integrator =
                            new LogicIntegrator.Builder(predictedData)
                                    .addConstraints(constraints)
                                    .sampleErrorRatesEstimates(methodNameParts[methodNameParts.length - 1].equals("SAMPLE"))
                                    .logProgress(detailedLog)
                                    .build();
                }
            } else {
                throw new IllegalArgumentException("The method name \"" + methodName + "\" is invalid.");
            }
            Integrator.ErrorRates errorRates = integrator.errorRates();
            Integrator.Data<Integrator.Data.PredictedInstance> integratedData = integrator.integratedData();
            results.add(evaluate(errorRates, integratedData, methodName));
        }
        logResults(new Results.Statistics(results), false);
    }

    private Integrator.ErrorRates computeSampleErrorRates(
            Integrator.Data<Integrator.Data.ObservedInstance> observedData,
            Integrator.Data<Integrator.Data.PredictedInstance> predictedData
    ) {
        List<Integrator.ErrorRates.Instance> instances = new ArrayList<>();
        for (Label label : labels) {
            int[] numberOfLabelInstances = { 0 };
            Map<Integer, Boolean> observations = new HashMap<>();
            observedData.stream()
                    .filter(instance -> instance.label().equals(label))
                    .forEach(instance -> observations.put(instance.instanceId(), instance.value()));
            predictedData.stream()
                    .filter(instance -> instance.label().equals(label))
                    .map(Integrator.Data.PredictedInstance::classifierId)
                    .distinct()
                    .forEach(classifierID -> {
                        Map<Integer, Double> predictions = new HashMap<>();
                        predictedData.stream()
                                .filter(instance -> observations.containsKey(instance.instanceId())
                                        && instance.label().equals(label)
                                        && instance.classifierId() == classifierID)
                                .forEach(instance -> predictions.put(instance.instanceId(), instance.value()));
                        numberOfLabelInstances[0] += predictions.size();
                        if (predictions.size() > 0)
                            instances.add(new Integrator.ErrorRates.Instance(
                                    label,
                                    classifierID,
                                    predictions.entrySet().stream()
                                            .filter(prediction -> (prediction.getValue() >= 0.5 && !observations.get(prediction.getKey()))
                                                    || (prediction.getValue() < 0.5 && observations.get(prediction.getKey())))
                                            .count() / (double) predictions.size())
                            );
                    });
            numberOfInstances.put(label, numberOfLabelInstances[0]);
        }
        return new Integrator.ErrorRates(instances);
    }

    private Results evaluate(Integrator.ErrorRates estimatedErrorRates,
                             Integrator.Data<Integrator.Data.PredictedInstance> integratedData,
                             String methodName) {
        List<Integrator.ErrorRates.Instance> estimatedErrorRatesList =
                estimatedErrorRates.stream()
                        .sorted((i1, i2) -> Integer.compare(i1.classifierID(), i2.classifierID()))
                        .sorted((i1, i2) -> i1.label().name().compareTo(i2.label().name()))
                        .collect(Collectors.toList());
        Map<Label, Map<Integer, Double>> errorRatesMap = new HashMap<>();
        Map<Label, Map<Integer, Double>> sampleErrorRatesMap = new HashMap<>();
        double mseError = 0.0;
        double madError = 0.0;
        double mseErrorWeighted = 0.0;
        double madErrorWeighted = 0.0;
        for (Integrator.ErrorRates.Instance instance : estimatedErrorRatesList) {
            if (!errorRatesMap.containsKey(instance.label())) {
                errorRatesMap.put(instance.label(), new HashMap<>());
                sampleErrorRatesMap.put(instance.label(), new HashMap<>());
            }
            errorRatesMap.get(instance.label()).put(instance.classifierID(), instance.errorRate());
            Optional<Integrator.ErrorRates.Instance> sampleErrorRateResultInstance =
                    sampleErrorRates.stream()
                            .filter(i -> i.label().equals(instance.label())
                                    && i.classifierID() == instance.classifierID())
                            .findFirst();
            double sampleErrorRate = 0.0;
            if (sampleErrorRateResultInstance.isPresent())
                sampleErrorRate = sampleErrorRateResultInstance.get().errorRate();
            sampleErrorRatesMap.get(instance.label()).put(instance.classifierID(), sampleErrorRate);
            double seError = (instance.errorRate() - sampleErrorRate) * (instance.errorRate() - sampleErrorRate);
            double adError = Math.abs(instance.errorRate() - sampleErrorRate);
            mseError += seError;
            madError += adError;
            mseErrorWeighted += seError / numberOfInstances.get(instance.label());
            madErrorWeighted += adError / numberOfInstances.get(instance.label());
        }
        mseError /= estimatedErrorRatesList.size();
        madError /= estimatedErrorRatesList.size();
        mseErrorWeighted /= labels.size();
        madErrorWeighted /= labels.size();
        Map<Integer, Map<Label, Boolean>> estimatedLabelsMap = new HashMap<>();
        Map<Integer, Map<Label, Boolean>> trueLabelsMap = new HashMap<>();
        double[] madErrorRank = {0.0};
        double[] madErrorRankWeighted = {0.0};
        double[] madLabel = {0.0};
        double[] madLabelWeighted = {0.0};
        integratedData.stream().map(Integrator.Data.PredictedInstance::label).distinct().forEach(label -> {
            List<Integer> rankedByErrorRate =
                    CollectionUtilities.sortByValue(errorRatesMap.get(label)).entrySet().stream()
                            .map(Map.Entry::getKey)
                            .collect(Collectors.toList());
            List<Integer> rankedBySampleErrorRate =
                    CollectionUtilities.sortByValue(sampleErrorRatesMap.get(label)).entrySet().stream()
                            .map(Map.Entry::getKey)
                            .collect(Collectors.toList());
            for (int i = 0; i < rankedByErrorRate.size(); i++) {
                double adErrorRank = Math.abs(rankedByErrorRate.get(i) - rankedBySampleErrorRate.get(i));
                madErrorRank[0] += adErrorRank;
                madErrorRankWeighted[0] += adErrorRank * numberOfInstances.get(label);
            }
            int[] adLabel = {0};
            int[] numberOfSamples = {0};
            Map<Integer, Boolean> observedInstancesMap = new HashMap<>();
            observedData.stream()
                    .filter(i -> i.label().equals(label))
                    .forEach(instance -> observedInstancesMap.put(instance.instanceId(), instance.value()));
            integratedData.stream()
                    .filter(i -> i.label().equals(label) && observedInstancesMap.containsKey(i.instanceId()))
                    .forEach(instance -> {
                        if ((instance.value() >= 0.5) != observedInstancesMap.get(instance.instanceId()))
                            adLabel[0]++;
                        numberOfSamples[0]++;
                        if (!estimatedLabelsMap.containsKey(instance.instanceId()))
                            estimatedLabelsMap.put(instance.instanceId(), new HashMap<>());
                        estimatedLabelsMap.get(instance.instanceId()).put(label, instance.value() >= 0.5);
                        if (!trueLabelsMap.containsKey(instance.instanceId()))
                            trueLabelsMap.put(instance.instanceId(), new HashMap<>());
                        trueLabelsMap.get(instance.instanceId())
                                .put(label, observedInstancesMap.get(instance.instanceId()));
                    });
            double accuracyLabel = adLabel[0] / (double) numberOfSamples[0];
            madLabel[0] += accuracyLabel;
            madLabelWeighted[0] += accuracyLabel / numberOfInstances.get(label);
        });
        madErrorRank[0] /= labels.size();
        madErrorRankWeighted[0] /= numberOfInstances.values().stream().mapToInt(n -> n).sum();
        madLabel[0] /= integratedData.stream().map(Integrator.Data.PredictedInstance::label).distinct().count();
        madLabelWeighted[0] /= labels.size();
        return new Results(methodName, errorRatesMap, sampleErrorRatesMap, estimatedLabelsMap, trueLabelsMap,
                           madErrorRank[0], mseError, madError,
                           madErrorRankWeighted[0], mseErrorWeighted, madErrorWeighted,
                           madLabel[0], madLabelWeighted[0]);
    }

    private void logResults(Results results, boolean detailed) {
        if (detailed) {
            logger.info("%15s - Results -              Label             | Classifier ID | " +
                                "Estimated Error Rate | Sample Error Rate |", results.methodName);
            for (Map.Entry<Label, Map<Integer, Double>> errorRatesEntry : results.errorRates.entrySet())
                for (Map.Entry<Integer, Double> errorRateEntry : errorRatesEntry.getValue().entrySet())
                    logger.info("%15s - Results - %30s | %13s | %20s | %17s |",
                                results.methodName,
                                errorRatesEntry.getKey(),
                                componentIDsMap.inverse().get(errorRateEntry.getKey()),
                                DECIMAL_FORMAT.format(errorRateEntry.getValue()),
                                DECIMAL_FORMAT.format(results.sampleErrorRates.get(errorRatesEntry.getKey()).get(errorRateEntry.getKey())));
            logger.info("                - Results - Error Rates RankMAD |     Error Rates MSE |     Error Rates MAD |          Labels MAD | " +
                                "Error Rates wRankMAD |    Error Rates wMSE |    Error Rates wMAD |         Labels wMAD |");
        }
        logger.info("%15s - Results - %19s | %15s | %15s | %14s | %20s | %16s | %16s | %13s |",
                    results.methodName,
                    RESULTS_DECIMAL_FORMAT.format(results.madErrorRank),
                    RESULTS_DECIMAL_FORMAT.format(results.mseError),
                    RESULTS_DECIMAL_FORMAT.format(results.madError),
                    RESULTS_DECIMAL_FORMAT.format(results.madLabel),
                    RESULTS_DECIMAL_FORMAT.format(results.madErrorRankWeighted),
                    RESULTS_DECIMAL_FORMAT.format(results.mseErrorWeighted),
                    RESULTS_DECIMAL_FORMAT.format(results.madErrorWeighted),
                    RESULTS_DECIMAL_FORMAT.format(results.madLabelWeighted));
    }

    private void logResults(Results.Statistics results, boolean includeTableHeader) {
        if (includeTableHeader)
            logger.info("                - Results - Error Rates RankMAD |     Error Rates MSE |     Error Rates MAD |          Labels MAD | " +
                                "Error Rates wRankMAD |    Error Rates wMSE |    Error Rates wMAD |         Labels wMAD |");
        logger.info("%15s - Results - %1.2e ± %1.2e | %1.2e ± %1.2e | %1.2e ± %1.2e | %1.2e ± %1.2e |  %1.2e ± %1.2e | %1.2e ± %1.2e | %1.2e ± %1.2e | %1.2e ± %1.2e |",
                    results.methodName,
                    results.madErrorRankMean,
                    results.madErrorRankVariance,
                    results.mseErrorMean,
                    results.mseErrorVariance,
                    results.madErrorMean,
                    results.madErrorVariance,
                    results.madLabelMean,
                    results.madLabelVariance,
                    results.madErrorRankWeightedMean,
                    results.madErrorRankWeightedVariance,
                    results.mseErrorWeightedMean,
                    results.mseErrorWeightedVariance,
                    results.madErrorWeightedMean,
                    results.madErrorWeightedVariance,
                    results.madLabelWeightedMean,
                    results.madLabelWeightedVariance);
//                    RESULTS_DECIMAL_FORMAT.format(results.madErrorRankMean),
//                    RESULTS_DECIMAL_FORMAT.format(results.madErrorRankVariance),
//                    RESULTS_DECIMAL_FORMAT.format(results.mseErrorMean),
//                    RESULTS_DECIMAL_FORMAT.format(results.mseErrorVariance),
//                    RESULTS_DECIMAL_FORMAT.format(results.madErrorMean),
//                    RESULTS_DECIMAL_FORMAT.format(results.madErrorVariance),
//                    RESULTS_DECIMAL_FORMAT.format(results.madLabelMean),
//                    RESULTS_DECIMAL_FORMAT.format(results.madLabelVariance),
//                    RESULTS_DECIMAL_FORMAT.format(results.madErrorRankWeightedMean),
//                    RESULTS_DECIMAL_FORMAT.format(results.madErrorRankWeightedVariance),
//                    RESULTS_DECIMAL_FORMAT.format(results.mseErrorWeightedMean),
//                    RESULTS_DECIMAL_FORMAT.format(results.mseErrorWeightedVariance),
//                    RESULTS_DECIMAL_FORMAT.format(results.madErrorWeightedMean),
//                    RESULTS_DECIMAL_FORMAT.format(results.madErrorWeightedVariance),
//                    RESULTS_DECIMAL_FORMAT.format(results.madLabelWeightedMean),
//                    RESULTS_DECIMAL_FORMAT.format(results.madLabelWeightedVariance));
    }

    private static class Results {
        private final String methodName;
        private final Map<Label, Map<Integer, Double>> errorRates;
        private final Map<Label, Map<Integer, Double>> sampleErrorRates;
        private final Map<Integer, Map<Label, Boolean>> estimatedLabels;
        private final Map<Integer, Map<Label, Boolean>> trueLabels;
        private final double madErrorRank;
        private final double mseError;
        private final double madError;
        private final double madErrorRankWeighted;
        private final double mseErrorWeighted;
        private final double madErrorWeighted;
        private final double madLabel;
        private final double madLabelWeighted;

        private Results(String methodName,
                        Map<Label, Map<Integer, Double>> errorRates,
                        Map<Label, Map<Integer, Double>> sampleErrorRates,
                        Map<Integer, Map<Label, Boolean>> estimatedLabels,
                        Map<Integer, Map<Label, Boolean>> trueLabels,
                        double madErrorRank,
                        double mseError,
                        double madError,
                        double madErrorRankWeighted,
                        double mseErrorWeighted,
                        double madErrorWeighted,
                        double madLabel,
                        double madLabelWeighted) {
            this.methodName = methodName;
            this.errorRates = errorRates;
            this.sampleErrorRates = sampleErrorRates;
            this.estimatedLabels = estimatedLabels;
            this.trueLabels = trueLabels;
            this.madErrorRank = madErrorRank;
            this.mseError = mseError;
            this.madError = madError;
            this.madErrorRankWeighted = madErrorRankWeighted;
            this.mseErrorWeighted = mseErrorWeighted;
            this.madErrorWeighted = madErrorWeighted;
            this.madLabel = madLabel;
            this.madLabelWeighted = madLabelWeighted;
        }

        private static class Statistics {
            private final String methodName;
            private final double madErrorRankMean;
            private final double madErrorRankVariance;
            private final double mseErrorMean;
            private final double mseErrorVariance;
            private final double madErrorMean;
            private final double madErrorVariance;
            private final double madErrorRankWeightedMean;
            private final double madErrorRankWeightedVariance;
            private final double mseErrorWeightedMean;
            private final double mseErrorWeightedVariance;
            private final double madErrorWeightedMean;
            private final double madErrorWeightedVariance;
            private final double madLabelMean;
            private final double madLabelVariance;
            private final double madLabelWeightedMean;
            private final double madLabelWeightedVariance;

            private Statistics(List<Results> results) {
                methodName = results.get(0).methodName;
                double[] madErrorRank = new double[results.size()];
                double[] mseError = new double[results.size()];
                double[] madError = new double[results.size()];
                double[] madErrorRankWeighted = new double[results.size()];
                double[] mseErrorWeighted = new double[results.size()];
                double[] madErrorWeighted = new double[results.size()];
                double[] madLabel = new double[results.size()];
                double[] madLabelWeighted = new double[results.size()];
                for (int index = 0; index < results.size(); index++) {
                    if (!results.get(index).methodName.equals(methodName))
                        throw new IllegalArgumentException("The method names of the provided results do not match.");
                    madErrorRank[index] = results.get(index).madErrorRank;
                    mseError[index] = results.get(index).mseError;
                    madError[index] = results.get(index).madError;
                    madErrorRankWeighted[index] = results.get(index).madErrorRankWeighted;
                    mseErrorWeighted[index] = results.get(index).mseErrorWeighted;
                    madErrorWeighted[index] = results.get(index).madErrorWeighted;
                    madLabel[index] = results.get(index).madLabel;
                    madLabelWeighted[index] = results.get(index).madLabelWeighted;
                }
                madErrorRankMean = StatisticsUtilities.mean(madErrorRank);
                madErrorRankVariance = StatisticsUtilities.variance(madErrorRank);
                mseErrorMean = StatisticsUtilities.mean(mseError);
                mseErrorVariance = StatisticsUtilities.variance(mseError);
                madErrorMean = StatisticsUtilities.mean(madError);
                madErrorVariance = StatisticsUtilities.variance(madError);
                madErrorRankWeightedMean = StatisticsUtilities.mean(madErrorRankWeighted);
                madErrorRankWeightedVariance = StatisticsUtilities.variance(madErrorRankWeighted);
                mseErrorWeightedMean = StatisticsUtilities.mean(mseErrorWeighted);
                mseErrorWeightedVariance = StatisticsUtilities.variance(mseErrorWeighted);
                madErrorWeightedMean = StatisticsUtilities.mean(madErrorWeighted);
                madErrorWeightedVariance = StatisticsUtilities.variance(madErrorWeighted);
                madLabelMean = StatisticsUtilities.mean(madLabel);
                madLabelVariance = StatisticsUtilities.variance(madLabel);
                madLabelWeightedMean = StatisticsUtilities.mean(madLabelWeighted);
                madLabelWeightedVariance = StatisticsUtilities.variance(madLabelWeighted);
            }
        }
    }

    private static InputData parseLabeledDataFromCSVFile(
            File directory,
            String separator,
            double[] classificationThresholds
    ) {
        List<Integrator.Data.PredictedInstance> predictedInstances = new ArrayList<>();
        List<Integrator.Data.ObservedInstance> observedInstances = new ArrayList<>();
        BiMap<String, Integer> instanceIDsMap = HashBiMap.create();
        BiMap<String, Integer> componentIDsMap = HashBiMap.create();
        for (File file : directory.listFiles()) {
            BufferedReader br = null;
            String line;
            try {
                br = new BufferedReader(new FileReader(file));
                br.readLine();
                Label label = new Label(file.getName());
                int lineNumber = 0;
                while ((line = br.readLine()) != null) {
                    String[] outputs = line.split(separator);
                    observedInstances.add(new Integrator.Data.ObservedInstance(lineNumber, label, !outputs[0].equals("0")));
                    for (int i = 1; i < outputs.length; i++) {
                        double value;
                        if (classificationThresholds == null)
                            value = Double.parseDouble(outputs[i]) >= 0.5 ? 1.0 : 0.0;
                        else if (classificationThresholds.length == 1)
                            value = Double.parseDouble(outputs[i]) >= classificationThresholds[0] ? 1.0 : 0.0;
                        else
                            value = Double.parseDouble(outputs[i]) >= classificationThresholds[i - 1] ? 1.0 : 0.0;
                        predictedInstances.add(new Integrator.Data.PredictedInstance(lineNumber, label, i - 1, value));
                        componentIDsMap.computeIfAbsent(String.valueOf(i - 1), key -> componentIDsMap.size());
                    }
                    instanceIDsMap.computeIfAbsent(String.valueOf(lineNumber), key -> instanceIDsMap.size());
                    lineNumber++;
                }
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                if (br != null) {
                    try {
                        br.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
        return new InputData(new Integrator.Data<>(predictedInstances),
                             new Integrator.Data<>(observedInstances),
                             instanceIDsMap,
                             componentIDsMap);
    }

    private static class InputData {
        private final Integrator.Data<Integrator.Data.PredictedInstance> predictedData;
        private final Integrator.Data<Integrator.Data.ObservedInstance> observedData;
        private final BiMap<String, Integer> instanceIDsMap;
        private final BiMap<String, Integer> componentIDsMap;

        private InputData(Integrator.Data<Integrator.Data.PredictedInstance> predictedData,
                          Integrator.Data<Integrator.Data.ObservedInstance> observedData,
                          BiMap<String, Integer> instanceIDsMap,
                          BiMap<String, Integer> componentIDsMap) {
            this.predictedData = predictedData;
            this.observedData = observedData;
            this.instanceIDsMap = instanceIDsMap;
            this.componentIDsMap = componentIDsMap;
        }
    }

    public static void main(String[] args) {
        InputData data = null;
        Set<Constraint> constraints = new HashSet<>();
        switch (args[1]) {
            case "NELL":
                constraints = DataSets.importConstraints(args[2] + "/constraints.txt");
                DataSets.NELLData nellData = DataSets.importNELLData(args[2]);
                BiMap<String, Integer> instanceIDsMap = HashBiMap.create();
                BiMap<String, Integer> componentIDsMap = HashBiMap.create();
                List<Integrator.Data.ObservedInstance> observedInstances = new ArrayList<>();
                List<Integrator.Data.PredictedInstance> predictedInstances = new ArrayList<>();
                for (DataSets.NELLData.Instance instance : nellData) {
                    int instanceID = instanceIDsMap.computeIfAbsent(instance.nounPhrase(), key -> instanceIDsMap.size());
                    int componentID = componentIDsMap.computeIfAbsent(instance.component(), key -> componentIDsMap.size());
                    if (instance.component().equals("KI"))
                        observedInstances.add(new Integrator.Data.ObservedInstance(instanceID,
                                                                                   new Label(instance.category()),
                                                                                   instance.probability() >= 0.5));
                    else
                        predictedInstances.add(new Integrator.Data.PredictedInstance(instanceID,
                                                                                     new Label(instance.category()),
                                                                                     componentID,
                                                                                     instance.probability()));
                }
                data = new InputData(new Integrator.Data<>(predictedInstances),
                                     new Integrator.Data<>(observedInstances),
                                     instanceIDsMap,
                                     componentIDsMap);
                break;
            case "ICML-2016":
                data = parseLabeledDataFromCSVFile(new File(args[2]), ",", new double[] { Double.parseDouble(args[3]) });
                break;
        }
        if (data != null) {
            ErrorEstimationExperiment experiment = new ErrorEstimationExperiment(
                    data.observedData,
                    data.predictedData,
                    constraints,
                    data.instanceIDsMap,
                    data.componentIDsMap
            );
            String[] methodNames = args[0].split(",");
            if (methodNames.length > 1)
                logger.info("                - Results - Error Rates RankMAD |     Error Rates MSE |     Error Rates MAD |          Labels MAD | " +
                                    "Error Rates wRankMAD |    Error Rates wMSE |    Error Rates wMAD |         Labels wMAD |");
            for (String methodName : methodNames)
                experiment.runExperiment(methodName, methodNames.length == 1);
        }
    }
}
