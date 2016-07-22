package makina.learn.classification.reflection;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import makina.learn.classification.Label;

import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class MajorityVoteIntegrator extends Integrator {
    private static final Logger logger = LogManager.getLogger("Classification / Majority Vote Integrator");

    private final Set<Label> labels;

    private boolean needsComputeIntegratedDataAndErrorRates = true;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends Integrator.AbstractBuilder<T> {
        private final Set<Label> labels = new HashSet<>();

        public AbstractBuilder(Data<Data.PredictedInstance> data) {
            super(data);
            extractLabelsSet();
        }

        private AbstractBuilder(String predictedDataFilename) {
            super(predictedDataFilename);
        }

        private void extractLabelsSet() {
            data.stream().map(Data.Instance::label).forEach(labels::add);
        }

        public MajorityVoteIntegrator build() {
            return new MajorityVoteIntegrator(this);
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(Data<Data.PredictedInstance> data) {
            super(data);
        }

        public Builder(String predictedDataFilename) {
            super(predictedDataFilename);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    private MajorityVoteIntegrator(AbstractBuilder<?> builder) {
        super(builder);
        labels = builder.labels;
    }

    @Override
    public ErrorRates errorRates(boolean forceComputation) {
        if (forceComputation)
            needsComputeIntegratedDataAndErrorRates = true;
        computeIntegratedDataAndErrorRates();
        return errorRates;
    }

    @Override
    public Data<Data.PredictedInstance> integratedData(boolean forceComputation) {
        if (forceComputation)
            needsComputeIntegratedDataAndErrorRates = true;
        computeIntegratedDataAndErrorRates();
        return integratedData;
    }

    private void computeIntegratedDataAndErrorRates() {
        if (!needsComputeIntegratedDataAndErrorRates)
            return;
        List<Data.PredictedInstance> integratedDataInstances = new ArrayList<>();
        List<ErrorRates.Instance> errorRatesInstances = new ArrayList<>();
        Map<Label, Map<Integer, Boolean>> integratedPredictions = new HashMap<>();
        for (Label label : labels) {
            integratedPredictions.put(label, new HashMap<>());
            Map<Integer, int[]> predictions = new HashMap<>();
            data.stream()
                    .filter(i -> i.label().equals(label))
                    .forEach(instance -> {
                        if (!predictions.containsKey(instance.id()))
                            predictions.put(instance.id(), new int[2]);
                        if (instance.value() >= 0.5)
                            predictions.get(instance.id())[0]++;
                        else
                            predictions.get(instance.id())[1]++;
                    });
            for (Map.Entry<Integer, int[]> prediction : predictions.entrySet())
                if (prediction.getValue()[0] >= prediction.getValue()[1]) {
                    integratedDataInstances.add(new Data.PredictedInstance(prediction.getKey(), label, -1, 1.0));
                    integratedPredictions.get(label).put(prediction.getKey(), true);
                } else {
                    integratedDataInstances.add(new Data.PredictedInstance(prediction.getKey(), label, -1, 0.0));
                    integratedPredictions.get(label).put(prediction.getKey(), false);
                }
            data.stream()
                    .filter(i -> i.label().equals(label))
                    .map(Data.PredictedInstance::functionId)
                    .distinct()
                    .forEach(classifierID -> {
                        int[] numberOfErrorSamples = new int[] { 0 };
                        int[] numberOfSamples = new int[] { 0 };
                        data.stream()
                                .filter(i -> i.label().equals(label) && i.functionId() == classifierID)
                                .forEach(instance -> {
                                    if ((instance.value() >= 0.5 && !integratedPredictions.get(label).get(instance.id()))
                                            || (instance.value() < 0.5 && integratedPredictions.get(label).get(instance.id())))
                                        numberOfErrorSamples[0]++;
                                    numberOfSamples[0]++;
                                });
                        errorRatesInstances.add(new ErrorRates.Instance(label, classifierID, numberOfErrorSamples[0] / (double) numberOfSamples[0]));
                    });
        }
        integratedData = new Data<>(integratedDataInstances);
        errorRates = new ErrorRates(errorRatesInstances);
        needsComputeIntegratedDataAndErrorRates = false;
    }
}
