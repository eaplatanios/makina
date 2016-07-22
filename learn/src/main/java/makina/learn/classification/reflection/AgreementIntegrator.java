package makina.learn.classification.reflection;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import makina.learn.classification.Label;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Handles the set up of the numerical optimization problem that needs to be solved in order to estimate error rates of
 * several approximations to a single function, by using only the agreement rates of those functions on an unlabeled set
 * of data.
 *
 * @author Emmanouil Antonios Platanios
 */
public final class AgreementIntegrator extends Integrator {
    private static final Logger logger = LogManager.getLogger("Classification / Agreement Integrator");

    private final BiMap<Integer, Integer> functionIdsMap = HashBiMap.create();
    private final Map<Label, AgreementRatesPowerSetVector> agreementRates = new HashMap<>();
    private final Map<Label, ErrorRatesPowerSetVector> sampleErrorRates = new HashMap<>();
    private final Map<Label, AgreementIntegratorOptimization> optimizationProblems = new HashMap<>();
    private final Map<Label, ErrorRatesPowerSetVector> errorRatesVectors = new HashMap<>();

    private final int highestOrder;

    private boolean needsComputeErrorRates = true;
    private boolean needsComputeIntegratedData = true;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends Integrator.AbstractBuilder<T> {
        private Data<Data.ObservedInstance> observedData = null;
        private int highestOrder = -1;
        private boolean onlyEvenCardinalitySubsetsAgreements = true;
        private AgreementIntegratorObjective objectiveFunctionType = AgreementIntegratorObjective.DEPENDENCY;
        private AgreementIntegratorInternalSolver internalSolver = AgreementIntegratorInternalSolver.IP_OPT;

        private AbstractBuilder(Data<Data.PredictedInstance> predictedData) {
            super(predictedData);
        }

        private AbstractBuilder(Data<Data.PredictedInstance> predictedData, Data<Data.ObservedInstance> observedData) {
            super(predictedData);
            this.observedData = observedData;
        }

        private AbstractBuilder(String predictedDataFilename) {
            super(predictedDataFilename);
        }

        private AbstractBuilder(String predictedDataFilename, String observedDataFilename) {
            super(predictedDataFilename);
            List<Data.ObservedInstance> observedInstances = new ArrayList<>();
            if(!loadObservedInstances(observedDataFilename, observedInstances))
                throw new RuntimeException("The observed integrator data could not be loaded from the provided file.");
            observedData = new Data<>(observedInstances);
        }

        public T highestOrder(int highestOrder) {
            this.highestOrder = highestOrder;
            return self();
        }

        public T onlyEvenCardinalitySubsetsAgreements(boolean onlyEvenCardinalitySubsetsAgreements) {
            this.onlyEvenCardinalitySubsetsAgreements = onlyEvenCardinalitySubsetsAgreements;
            return self();
        }

        public T objectiveFunctionType(AgreementIntegratorObjective objectiveFunctionType) {
            this.objectiveFunctionType = objectiveFunctionType;
            return self();
        }

        public T optimizationSolverType(AgreementIntegratorInternalSolver optimizationSolverType) {
            this.internalSolver = optimizationSolverType;
            return self();
        }

        public AgreementIntegrator build() {
            return new AgreementIntegrator(this);
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(Data<Data.PredictedInstance> predictedData) {
            super(predictedData);
        }

        public Builder(Data<Data.PredictedInstance> predictedData, Data<Data.ObservedInstance> observedData) {
            super(predictedData, observedData);
        }

        public Builder(String predictedDataFilename) {
            super(predictedDataFilename);
        }

        public Builder(String predictedDataFilename, String observedDataFilename) {
            super(predictedDataFilename, observedDataFilename);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    /**
     * Sets up the optimization problem that needs to be solved for estimating the error rates, using the settings
     * included in the provided {@link AgreementIntegrator.Builder} object, and also initializes the numerical
     * optimization solver that is used to solve it.
     *
     * @param   builder The builder object containing the settings to be used.
     */
    private AgreementIntegrator(AbstractBuilder<?> builder) {
        super(builder);
        data.stream()
                .map(Data.PredictedInstance::functionId)
                .distinct()
                .forEach(id -> functionIdsMap.put(id, functionIdsMap.size()));
        highestOrder = builder.highestOrder == -1 ? functionIdsMap.size() : builder.highestOrder;
        data.stream().map(Data.Instance::label).distinct().forEach(label -> {
            AgreementRatesPowerSetVector agreementRatesVector = new AgreementRatesPowerSetVector(
                    functionIdsMap.size(),
                    highestOrder,
                    data,
                    builder.onlyEvenCardinalitySubsetsAgreements,
                    functionIdsMap
            );
            ErrorRatesPowerSetVector errorRatesVector = new ErrorRatesPowerSetVector(
                    functionIdsMap.size(), highestOrder, 0.25
            );
            agreementRates.put(label, agreementRatesVector);
            optimizationProblems.put(
                    label,
                    builder.internalSolver.buildOptimizationProblem(
                            functionIdsMap.size(),
                            highestOrder,
                            errorRatesVector,
                            agreementRatesVector,
                            builder.objectiveFunctionType
                    )
            );
            if (builder.observedData != null) {
                ErrorRatesPowerSetVector sampleErrorRatesVector = new ErrorRatesPowerSetVector(
                        functionIdsMap.size(),
                        builder.highestOrder,
                        builder.data,
                        builder.observedData,
                        functionIdsMap
                );
                sampleErrorRates.put(label, sampleErrorRatesVector);
            }
            errorRatesVectors.put(label, errorRatesVector);
        });
    }

    public Map<Label, AgreementRatesPowerSetVector> agreementRates() {
        return agreementRates;
    }

    public Map<Label, ErrorRatesPowerSetVector> sampleErrorRates() {
        return sampleErrorRates;
    }

    public Map<Label, ErrorRatesPowerSetVector> errorRatesVectors() {
        return errorRatesVectors;
    }

    @Override
    public ErrorRates errorRates(boolean forceComputation) {
        if (forceComputation)
            needsComputeErrorRates = true;
        computeErrorRates();
        return errorRates;
    }

    private void computeErrorRates() {
        if (!needsComputeErrorRates)
            return;
        List<ErrorRates.Instance> errorRatesInstances = new ArrayList<>();
        for (Map.Entry<Label, AgreementIntegratorOptimization> problemEntry : optimizationProblems.entrySet()) {
            double[] labelErrorRates = problemEntry.getValue().solve();
            errorRatesVectors.get(problemEntry.getKey()).array = labelErrorRates;
            for (int index = 0; index < functionIdsMap.size(); index++)
                errorRatesInstances.add(new ErrorRates.Instance(problemEntry.getKey(),
                                                                functionIdsMap.inverse().get(index),
                                                                labelErrorRates[index]));
        }
        errorRates = new ErrorRates(errorRatesInstances);
        needsComputeErrorRates = false;
    }

    @Override
    public Data<Data.PredictedInstance> integratedData(boolean forceComputation) {
        if (forceComputation)
            needsComputeIntegratedData = true;
        computeIntegratedData();
        return integratedData;
    }

    private void computeIntegratedData() {
        if (!needsComputeIntegratedData)
            return;
        List<Data.PredictedInstance> integratedDataInstances = new ArrayList<>();
        for (Label label : optimizationProblems.keySet()) {
            Map<Integer, Double> errorRatesMap = new HashMap<>();
            errorRates.stream()
                    .filter(instance -> instance.label().equals(label))
                    .forEach(instance -> errorRatesMap.put(instance.functionId(), instance.errorRate()));
            Map<Integer, double[]> predictions = new HashMap<>();
            data.stream()
                    .filter(i -> i.label().equals(label))
                    .forEach(instance -> {
                        if (!predictions.containsKey(instance.id()))
                            predictions.put(instance.id(), new double[2]);
                        predictions.get(instance.id())[0] += instance.value() * (1 - errorRatesMap.get(instance.functionId()));
                        predictions.get(instance.id())[1] += (1 - instance.value()) * (1 - errorRatesMap.get(instance.functionId()));
                    });
            integratedDataInstances.addAll(
                    predictions.entrySet().stream()
                            .map(prediction -> new Data.PredictedInstance(
                                    prediction.getKey(),
                                    label,
                                    prediction.getValue()[0] / (prediction.getValue()[0] + prediction.getValue()[1])
                            ))
                            .collect(Collectors.toList()));
        }
        integratedData = new Data<>(integratedDataInstances);
        needsComputeIntegratedData = false;
    }
}
