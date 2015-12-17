package org.platanios.learn.classification.active;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.data.DataInstance;
import org.platanios.learn.math.matrix.Vector;

import java.util.*;

/**
 * TODO: Change the code so that we have one data set over everything with a List<Label> as its label type, or something
 * like that.
 *
 * @author Emmanouil Antonios Platanios
 */
public class Learning {
    private static final Logger logger = LogManager.getLogger("Classification / Active Learner");

    protected final Set<Label> labels;
    protected final ActiveLearningMethod activeLearningMethod;

    protected Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>> {
        /** A self-reference to this builder class. This is basically part of a small "hack" so that we can have
         * inheritable builder classes. */
        protected abstract T self();

        protected Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet;

        protected Set<Label> labels = new HashSet<>();
        protected ActiveLearningMethod activeLearningMethod = ActiveLearningMethod.UNCERTAINTY_HEURISTIC;

        protected AbstractBuilder(Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet) {
            this.dataSet = dataSet;
        }

        public T addLabel(Label label) {
            labels.add(label);
            return self();
        }

        public T addLabels(Set<Label> labels) {
            this.labels.addAll(labels);
            return self();
        }

        public T activeLearningMethod(ActiveLearningMethod activeLearningMethod) {
            this.activeLearningMethod = activeLearningMethod;
            return self();
        }

        public Learning build() {
            return new Learning(this);
        }
    }

    /**
     * The builder class for this abstract class. This is basically part of a small "hack" so that we can have
     * inheritable builder classes.
     */
    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet) {
            super(dataSet);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }
    }

    protected Learning(AbstractBuilder<?> builder) {
        labels = builder.labels;
        activeLearningMethod = builder.activeLearningMethod;
        dataSet = builder.dataSet;
    }

    public Map<DataInstance<Vector>, Map<Label, Boolean>> getDataSet() {
        return dataSet;
    }

    public Map<Label, Boolean> getLabels(DataInstance<Vector> instance) {
        return dataSet.get(instance);
    }

    public int getNumberOfLabeledInstances(Label label) {
        int numberOfLabeledInstances = 0;
        for (Map.Entry<DataInstance<Vector>, Map<Label, Boolean>> instanceEntry : dataSet.entrySet())
            if (instanceEntry.getValue().keySet().contains(label))
                numberOfLabeledInstances++;
        return numberOfLabeledInstances;
    }

    public int getNumberOfLabeledInstances() {
        int numberOfInstances = 0;
        for (Label label : labels)
            numberOfInstances += getNumberOfLabeledInstances(label);
        return numberOfInstances;
    }

    public InstanceToLabel pickInstanceToLabel(Map<DataInstance<Vector>, Map<Label, Double>> dataSet) {
        return activeLearningMethod.pickInstanceToLabel(this, dataSet);
    }

    public List<InstanceToLabel> pickInstancesToLabel(Map<DataInstance<Vector>, Map<Label, Double>> dataSet,
                                                      int numberOfInstancesToPick) {
        return activeLearningMethod.pickInstancesToLabel(this, dataSet, numberOfInstancesToPick);
    }

    public void labelInstance(InstanceToLabel instance, Boolean label) {
        if (!dataSet.containsKey(instance.instance))
            dataSet.put(instance.instance, new HashMap<>());
        dataSet.get(instance.instance).put(instance.label, label);
    }

    public void labelInstances(Map<InstanceToLabel, Boolean> instancesToLabel) {
        for (Map.Entry<InstanceToLabel, Boolean> instance : instancesToLabel.entrySet())
            labelInstance(instance.getKey(), instance.getValue());
    }

    public static class InstanceToLabel {
        private final Label label;
        private final DataInstance<Vector> instance;

        public InstanceToLabel(Label label, DataInstance<Vector> instance) {
            this.label = label;
            this.instance = instance;
        }

        public Label getLabel() {
            return label;
        }

        public DataInstance<Vector> getInstance() {
            return instance;
        }
    }
}
