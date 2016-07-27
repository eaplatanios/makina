package makina.experiment.classification.reflection;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import makina.learn.classification.reflection.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import makina.experiment.data.DataSets;
import makina.learn.classification.Label;
import makina.learn.classification.constraint.Constraint;
import makina.learn.evaluation.PrecisionRecall;
import makina.math.StatisticsUtilities;
import makina.utilities.CollectionUtilities;

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
public class IntegratorExperiment {
    private static final DecimalFormat DECIMAL_FORMAT = new DecimalFormat("0.0000000000E0");
    private static final DecimalFormat RESULTS_DECIMAL_FORMAT = new DecimalFormat("0.00E0");
    private static final Logger logger = LogManager.getFormatterLogger("Classification / Reflection / Integrator Experiment");

    private final Map<Label, Integer> numberOfInstances = new HashMap<>();

    private final Integrator.ErrorRates sampleErrorRates;
    private final Integrator.Data<Integrator.Data.ObservedInstance> observedData;
    private final Integrator.Data<Integrator.Data.PredictedInstance> predictedData;
    private final Set<Label> labels;
    private final Set<Constraint> constraints;
//    private final BiMap<String, Integer> instanceIds;
    private final BiMap<String, Integer> componentIds;

    private IntegratorExperiment(Integrator.Data<Integrator.Data.ObservedInstance> observedData,
                                 Integrator.Data<Integrator.Data.PredictedInstance> predictedData,
                                 Set<Constraint> constraints,
                                 BiMap<String, Integer> instanceIds,
                                 BiMap<String, Integer> componentIds) {
        labels = predictedData.stream().map(Integrator.Data.Instance::label).collect(Collectors.toSet());
        sampleErrorRates = computeSampleErrorRates(observedData, predictedData);
        this.observedData = observedData;
        this.predictedData = predictedData;
        this.constraints = constraints;
//        this.instanceIds = instanceIds;
        this.componentIds = componentIds;
        logger.info("Total number of instances: " + instanceIds.size());
        logger.info("Total number of labels: " + labels.size());
        logger.info("Total number of function approximations: " + componentIds.size());
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
                    boolean sampleEstimates = methodNameParts[methodNameParts.length - 1].equals("SAMPLE");
                    integrator =
                            new LogicIntegrator.Builder(predictedData)
                                    .addConstraints(constraints)
                                    .sampleErrorRatesEstimates(sampleEstimates)
                                    .logProgress(detailedLog)
                                    .build();
                }
            } else {
                throw new IllegalArgumentException("The method name \"" + methodName + "\" is invalid.");
            }
            Integrator.ErrorRates errorRates = integrator.errorRates();
            Integrator.Data<Integrator.Data.PredictedInstance> integratedData = integrator.integratedData();
            Results result = evaluate(errorRates, integratedData, methodName);
            results.add(result);
            if (detailedLog)
                logResults(result, true);
        }
        logResults(new Results.Statistics(results), detailedLog);
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
                    .forEach(instance -> observations.put(instance.id(), instance.value()));
            predictedData.stream()
                    .filter(instance -> instance.label().equals(label))
                    .map(Integrator.Data.PredictedInstance::functionId)
                    .distinct()
                    .forEach(classifierID -> {
                        Map<Integer, Double> predictions = new HashMap<>();
                        predictedData.stream()
                                .filter(instance -> observations.containsKey(instance.id())
                                        && instance.label().equals(label)
                                        && instance.functionId() == classifierID)
                                .forEach(instance -> predictions.put(instance.id(), instance.value()));
                        numberOfLabelInstances[0] += predictions.size();
                        if (predictions.size() > 0)
                            instances.add(new Integrator.ErrorRates.Instance(
                                    label,
                                    classifierID,
                                    predictions.entrySet().stream()
                                            .filter(p -> (p.getValue() >= 0.5 && !observations.get(p.getKey()))
                                                    || (p.getValue() < 0.5 && observations.get(p.getKey())))
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
                        .sorted((i1, i2) -> Integer.compare(i1.functionId(), i2.functionId()))
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
            errorRatesMap.get(instance.label()).put(instance.functionId(), instance.value());
            Optional<Integrator.ErrorRates.Instance> sampleErrorRateResultInstance =
                    sampleErrorRates.stream()
                            .filter(i -> i.label().equals(instance.label()) && i.functionId() == instance.functionId())
                            .findFirst();
            double sampleErrorRate = 0.0;
            if (sampleErrorRateResultInstance.isPresent())
                sampleErrorRate = sampleErrorRateResultInstance.get().value();
            sampleErrorRatesMap.get(instance.label()).put(instance.functionId(), sampleErrorRate);
            double seError = (instance.value() - sampleErrorRate) * (instance.value() - sampleErrorRate);
            double adError = Math.abs(instance.value() - sampleErrorRate);
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
        double[] aucLabel = {0.0};
        double[] madLabel = {0.0};
        double[] madHardLabel = {0.0};
        double[] aucLabelWeighted = {0.0};
        double[] madLabelWeighted = {0.0};
        double[] madHardLabelWeighted = {0.0};
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
            double[] adLabel = {0};
            double[] adHardLabel = {0};
            int[] numberOfSamples = {0};
            Map<Integer, Boolean> observedInstances = new HashMap<>();
            observedData.stream()
                    .filter(i -> i.label().equals(label))
                    .forEach(instance -> observedInstances.put(instance.id(), instance.value()));
            // Compute AUC for labels evaluation
            List<Boolean> observedLabels = new ArrayList<>();
            List<Double> predictions = new ArrayList<>();
            integratedData.stream()
                    .filter(i -> i.label().equals(label) && observedInstances.containsKey(i.id()))
                    .sorted((i1, i2) -> i1.value() - i2.value() == 0.0 ? 0 : i1.value() - i2.value() > 0.0 ? -1 : 1)
                    .forEach(i -> {
                        observedLabels.add(observedInstances.get(i.id()));
                        predictions.add(i.value());
                    });
            double areaUnderTheCurve = PrecisionRecall.areaUnderTheCurve(observedLabels, predictions);
            aucLabel[0] += areaUnderTheCurve;
            aucLabelWeighted[0] += areaUnderTheCurve / numberOfInstances.get(label);
            // Compute MAD for labels evaluation
            integratedData.stream()
                    .filter(i -> i.label().equals(label) && observedInstances.containsKey(i.id()))
                    .forEach(instance -> {
                        adLabel[0] += Math.abs(instance.value() - (observedInstances.get(instance.id()) ? 1.0 : 0.0));
                        if ((instance.value() >= 0.5) != observedInstances.get(instance.id()))
                            adHardLabel[0]++;
                        numberOfSamples[0]++;
                        if (!estimatedLabelsMap.containsKey(instance.id()))
                            estimatedLabelsMap.put(instance.id(), new HashMap<>());
                        estimatedLabelsMap.get(instance.id()).put(label, instance.value() >= 0.5);
                        if (!trueLabelsMap.containsKey(instance.id()))
                            trueLabelsMap.put(instance.id(), new HashMap<>());
                        trueLabelsMap.get(instance.id())
                                .put(label, observedInstances.get(instance.id()));
                    });
            double accuracyLabel = adLabel[0] / (double) numberOfSamples[0];
            double accuracyHardLabel = adHardLabel[0] / (double) numberOfSamples[0];
            madLabel[0] += accuracyLabel;
            madHardLabel[0] += accuracyHardLabel;
            madLabelWeighted[0] += accuracyLabel / numberOfInstances.get(label);
            madHardLabelWeighted[0] += accuracyHardLabel / numberOfInstances.get(label);
        });
        madErrorRank[0] /= labels.size();
        madErrorRankWeighted[0] /= numberOfInstances.values().stream().mapToInt(n -> n).sum();
        aucLabel[0] /= labels.size();
        madLabel[0] /= labels.size();
        madHardLabel[0] /= labels.size();
        aucLabelWeighted[0] /= labels.size();
        madLabelWeighted[0] /= labels.size();
        madHardLabelWeighted[0] /= labels.size();
        return new Results(methodName, errorRatesMap, sampleErrorRatesMap, // , estimatedLabelsMap, trueLabelsMap,
                           madErrorRank[0], mseError, madError,
                           madErrorRankWeighted[0], mseErrorWeighted, madErrorWeighted,
                           aucLabel[0], madLabel[0], madHardLabel[0],
                           aucLabelWeighted[0], madLabelWeighted[0], madHardLabelWeighted[0]);
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
                                componentIds.inverse().get(errorRateEntry.getKey()),
                                DECIMAL_FORMAT.format(errorRateEntry.getValue()),
                                DECIMAL_FORMAT.format(results.sampleErrorRates.get(errorRatesEntry.getKey()).get(errorRateEntry.getKey())));
            logger.info("                - Results - Error Rates RankMAD |     Error Rates MSE |     Error Rates MAD | " +
                                "         Labels AUC |          Labels MAD |     Hard Labels MAD | " +
                                "Error Rates wRankMAD |    Error Rates wMSE |    Error Rates wMAD | " +
                                "        Labels wAUC |         Labels wMAD |    Hard Labels wMAD |");
        }
        logger.info("%15s - Results - %19s | %19s | %19s | %19s | %19s | %19s | %20s | %19s | %19s | %19s | %19s | %19s |",
                    results.methodName,
                    RESULTS_DECIMAL_FORMAT.format(results.madErrorRank),
                    RESULTS_DECIMAL_FORMAT.format(results.mseError),
                    RESULTS_DECIMAL_FORMAT.format(results.madError),
                    RESULTS_DECIMAL_FORMAT.format(results.aucLabel),
                    RESULTS_DECIMAL_FORMAT.format(results.madLabel),
                    RESULTS_DECIMAL_FORMAT.format(results.madHardLabel),
                    RESULTS_DECIMAL_FORMAT.format(results.madErrorRankWeighted),
                    RESULTS_DECIMAL_FORMAT.format(results.mseErrorWeighted),
                    RESULTS_DECIMAL_FORMAT.format(results.madErrorWeighted),
                    RESULTS_DECIMAL_FORMAT.format(results.aucLabelWeighted),
                    RESULTS_DECIMAL_FORMAT.format(results.madLabelWeighted),
                    RESULTS_DECIMAL_FORMAT.format(results.madHardLabelWeighted));
    }

    private void logResults(Results.Statistics results, boolean includeTableHeader) {
        if (includeTableHeader)
            logger.info("                - Results -" +
                                " Error Rates RankMAD |     Error Rates MSE |     Error Rates MAD |" +
                                "          Labels AUC |          Labels MAD |     Hard Labels MAD |" +
                                " Error Rates wRankMAD |    Error Rates wMSE |    Error Rates wMAD |" +
                                "         Labels wAUC |         Labels wMAD |    Hard Labels wMAD |");
        logger.info("%15s - Results - %1.2e ± %1.2e | %1.2e ± %1.2e | %1.2e ± %1.2e |" +
                            " %1.2e ± %1.2e | %1.2e ± %1.2e | %1.2e ± %1.2e |" +
                            "  %1.2e ± %1.2e | %1.2e ± %1.2e | %1.2e ± %1.2e |" +
                            " %1.2e ± %1.2e | %1.2e ± %1.2e | %1.2e ± %1.2e |",
                    results.methodName,
                    results.madErrorRankMean,
                    results.madErrorRankVariance,
                    results.mseErrorMean,
                    results.mseErrorVariance,
                    results.madErrorMean,
                    results.madErrorVariance,
                    results.aucLabelMean,
                    results.aucLabelVariance,
                    results.madLabelMean,
                    results.madLabelVariance,
                    results.madHardLabelMean,
                    results.madHardLabelVariance,
                    results.madErrorRankWeightedMean,
                    results.madErrorRankWeightedVariance,
                    results.mseErrorWeightedMean,
                    results.mseErrorWeightedVariance,
                    results.madErrorWeightedMean,
                    results.madErrorWeightedVariance,
                    results.aucLabelWeightedMean,
                    results.aucLabelWeightedVariance,
                    results.madLabelWeightedMean,
                    results.madLabelWeightedVariance,
                    results.madHardLabelWeightedMean,
                    results.madHardLabelWeightedVariance);
    }

    private static class Results {
        private final String methodName;
        private final Map<Label, Map<Integer, Double>> errorRates;
        private final Map<Label, Map<Integer, Double>> sampleErrorRates;
//        private final Map<Integer, Map<Label, Boolean>> estimatedLabels;
//        private final Map<Integer, Map<Label, Boolean>> trueLabels;
        private final double madErrorRank;
        private final double mseError;
        private final double madError;
        private final double madErrorRankWeighted;
        private final double mseErrorWeighted;
        private final double madErrorWeighted;
        private final double aucLabel;
        private final double madLabel;
        private final double madHardLabel;
        private final double aucLabelWeighted;
        private final double madLabelWeighted;
        private final double madHardLabelWeighted;

        private Results(String methodName,
                        Map<Label, Map<Integer, Double>> errorRates,
                        Map<Label, Map<Integer, Double>> sampleErrorRates,
//                        Map<Integer, Map<Label, Boolean>> estimatedLabels,
//                        Map<Integer, Map<Label, Boolean>> trueLabels,
                        double madErrorRank,
                        double mseError,
                        double madError,
                        double madErrorRankWeighted,
                        double mseErrorWeighted,
                        double madErrorWeighted,
                        double aucLabel,
                        double madLabel,
                        double madHardLabel,
                        double aucLabelWeighted,
                        double madLabelWeighted,
                        double madHardLabelWeighted) {
            this.methodName = methodName;
            this.errorRates = errorRates;
            this.sampleErrorRates = sampleErrorRates;
//            this.estimatedLabels = estimatedLabels;
//            this.trueLabels = trueLabels;
            this.madErrorRank = madErrorRank;
            this.mseError = mseError;
            this.madError = madError;
            this.madErrorRankWeighted = madErrorRankWeighted;
            this.mseErrorWeighted = mseErrorWeighted;
            this.madErrorWeighted = madErrorWeighted;
            this.aucLabel = aucLabel;
            this.madLabel = madLabel;
            this.madHardLabel = madHardLabel;
            this.aucLabelWeighted = aucLabelWeighted;
            this.madLabelWeighted = madLabelWeighted;
            this.madHardLabelWeighted = madHardLabelWeighted;
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
            private final double aucLabelMean;
            private final double aucLabelVariance;
            private final double madLabelMean;
            private final double madLabelVariance;
            private final double madHardLabelMean;
            private final double madHardLabelVariance;
            private final double aucLabelWeightedMean;
            private final double aucLabelWeightedVariance;
            private final double madLabelWeightedMean;
            private final double madLabelWeightedVariance;
            private final double madHardLabelWeightedMean;
            private final double madHardLabelWeightedVariance;

            private Statistics(List<Results> results) {
                methodName = results.get(0).methodName;
                double[] madErrorRank = new double[results.size()];
                double[] mseError = new double[results.size()];
                double[] madError = new double[results.size()];
                double[] madErrorRankWeighted = new double[results.size()];
                double[] mseErrorWeighted = new double[results.size()];
                double[] madErrorWeighted = new double[results.size()];
                double[] aucLabel = new double[results.size()];
                double[] madLabel = new double[results.size()];
                double[] madHardLabel = new double[results.size()];
                double[] aucLabelWeighted = new double[results.size()];
                double[] madLabelWeighted = new double[results.size()];
                double[] madHardLabelWeighted = new double[results.size()];
                for (int index = 0; index < results.size(); index++) {
                    if (!results.get(index).methodName.equals(methodName))
                        throw new IllegalArgumentException("The method names of the provided results do not match.");
                    madErrorRank[index] = results.get(index).madErrorRank;
                    mseError[index] = results.get(index).mseError;
                    madError[index] = results.get(index).madError;
                    madErrorRankWeighted[index] = results.get(index).madErrorRankWeighted;
                    mseErrorWeighted[index] = results.get(index).mseErrorWeighted;
                    madErrorWeighted[index] = results.get(index).madErrorWeighted;
                    aucLabel[index] = results.get(index).aucLabel;
                    madLabel[index] = results.get(index).madLabel;
                    madHardLabel[index] = results.get(index).madHardLabel;
                    aucLabelWeighted[index] = results.get(index).aucLabelWeighted;
                    madLabelWeighted[index] = results.get(index).madLabelWeighted;
                    madHardLabelWeighted[index] = results.get(index).madHardLabelWeighted;
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
                aucLabelMean = StatisticsUtilities.mean(aucLabel);
                aucLabelVariance = StatisticsUtilities.variance(aucLabel);
                madLabelMean = StatisticsUtilities.mean(madLabel);
                madLabelVariance = StatisticsUtilities.variance(madLabel);
                madHardLabelMean = StatisticsUtilities.mean(madHardLabel);
                madHardLabelVariance = StatisticsUtilities.variance(madHardLabel);
                aucLabelWeightedMean = StatisticsUtilities.mean(aucLabelWeighted);
                aucLabelWeightedVariance = StatisticsUtilities.variance(aucLabelWeighted);
                madLabelWeightedMean = StatisticsUtilities.mean(madLabelWeighted);
                madLabelWeightedVariance = StatisticsUtilities.variance(madLabelWeighted);
                madHardLabelWeightedMean = StatisticsUtilities.mean(madHardLabelWeighted);
                madHardLabelWeightedVariance = StatisticsUtilities.variance(madHardLabelWeighted);
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
        File[] files = directory.listFiles();
        if (files == null)
            return null;
        for (File file : files) {
            BufferedReader br = null;
            String line;
            try {
                br = new BufferedReader(new FileReader(file));
                br.readLine();
                Label label = new Label(file.getName());
                int lineNumber = 0;
                while ((line = br.readLine()) != null) {
                    String[] outputs = line.split(separator);
                    observedInstances.add(new Integrator.Data.ObservedInstance(lineNumber,
                                                                               label,
                                                                               !outputs[0].equals("0")));
                    for (int i = 1; i < outputs.length; i++) {
                        double value;
                        if (classificationThresholds == null)
                            value = Double.parseDouble(outputs[i]);
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
        private final BiMap<String, Integer> instanceIds;
        private final BiMap<String, Integer> componentIds;

        private InputData(Integrator.Data<Integrator.Data.PredictedInstance> predictedData,
                          Integrator.Data<Integrator.Data.ObservedInstance> observedData,
                          BiMap<String, Integer> instanceIds,
                          BiMap<String, Integer> componentIds) {
            this.predictedData = predictedData;
            this.observedData = observedData;
            this.instanceIds = instanceIds;
            this.componentIds = componentIds;
        }
    }

    public static void main(String[] args) {
        boolean softPredictions = !args[1].equals("0");
        InputData data = null;
        Set<Constraint> constraints = new HashSet<>();
        switch (args[2]) {
            case "NELL":
                double threshold = Double.parseDouble(args[4]);
                constraints = DataSets.importConstraints(args[3] + "/constraints.txt");
                DataSets.NELLData nellData = DataSets.importNELLData(args[3]);
                BiMap<String, Integer> instanceIds = HashBiMap.create();
                BiMap<String, Integer> componentIds = HashBiMap.create();
                List<Integrator.Data.ObservedInstance> observedInstances = new ArrayList<>();
                List<Integrator.Data.PredictedInstance> predictedInstances = new ArrayList<>();
                for (DataSets.NELLData.Instance instance : nellData) {
                    int instanceID = instanceIds.computeIfAbsent(instance.nounPhrase(), key -> instanceIds.size());
                    int componentID = componentIds.computeIfAbsent(instance.component(), key -> componentIds.size());
                    if (instance.component().equals("KI"))
                        observedInstances.add(new Integrator.Data.ObservedInstance(instanceID,
                                                                                   new Label(instance.category()),
                                                                                   instance.probability() >= threshold));
                    else
                        predictedInstances.add(new Integrator.Data.PredictedInstance(
                                instanceID,
                                new Label(instance.category()),
                                componentID,
                                softPredictions ? instance.probability() : (instance.probability() >= 0.5 ? 1.0 : 0.0))
                        );
                }
                data = new InputData(new Integrator.Data<>(predictedInstances),
                                     new Integrator.Data<>(observedInstances),
                                     instanceIds,
                                     componentIds);
                break;
            case "ICML-2016":
                data = parseLabeledDataFromCSVFile(new File(args[3]), ",",
                                                   softPredictions ? null : new double[]{Double.parseDouble(args[4])});
                break;
        }
        if (data != null) {
            IntegratorExperiment experiment = new IntegratorExperiment(
                    data.observedData,
                    data.predictedData,
                    constraints,
                    data.instanceIds,
                    data.componentIds
            );
            String[] methodNames = args[0].split(",");
            if (methodNames.length > 1)
                logger.info("                - Results -" +
                                    " Error Rates RankMAD |     Error Rates MSE |     Error Rates MAD |" +
                                    "          Labels AUC |          Labels MAD |     Hard Labels MAD |" +
                                    " Error Rates wRankMAD |    Error Rates wMSE |    Error Rates wMAD |" +
                                    "         Labels wAUC |         Labels wMAD |    Hard Labels wMAD |");
            for (String methodName : methodNames)
                experiment.runExperiment(methodName, methodNames.length == 1);
        }
    }
}
