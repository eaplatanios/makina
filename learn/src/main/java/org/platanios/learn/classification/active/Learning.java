package org.platanios.learn.classification.active;

import com.google.common.base.Objects;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.classification.Label;
import org.platanios.learn.data.DataInstance;
import org.platanios.learn.math.matrix.Vector;

import java.util.*;
import java.util.function.Function;

/**
 * TODO: Change the code so that we have one data set over everything with a List<Label> as its label type, or something
 * like that.
 *
 * @author Emmanouil Antonios Platanios
 */
public class Learning {
    private static final Logger logger = LogManager.getLogger("Classification / Active Learner");

    protected final Function<InstanceToLabel, Double> probabilityFunction;
    protected final Set<Label> labels;
    protected final ScoringFunction scoringFunction;

    protected final TreeSet<InstanceToLabel> instancesToLabel = new TreeSet<>(
            Collections.reverseOrder(Comparator.comparing(instance -> instance.informationGainHeuristicValue))
    );

    protected Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>> {
        /** A self-reference to this builder class. This is basically part of a small "hack" so that we can have
         * inheritable builder classes. */
        protected abstract T self();

        protected final Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet;
        protected final Function<InstanceToLabel, Double> probabilityFunction;

        protected Set<Label> labels = new HashSet<>();
        protected ScoringFunction scoringFunction = new EntropyScoringFunction();

        protected AbstractBuilder(Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet,
                                  Function<InstanceToLabel, Double> probabilityFunction) {
            this.dataSet = dataSet;
            this.probabilityFunction = probabilityFunction;
        }

        public T addLabel(Label label) {
            labels.add(label);
            return self();
        }

        public T addLabels(Set<Label> labels) {
            this.labels.addAll(labels);
            return self();
        }

        public T activeLearningMethod(ScoringFunction scoringFunction) {
            this.scoringFunction = scoringFunction;
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
        public Builder(Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet,
                       Function<InstanceToLabel, Double> probabilityFunction) {
            super(dataSet, probabilityFunction);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }
    }

    protected Learning(AbstractBuilder<?> builder) {
        probabilityFunction = builder.probabilityFunction;
        labels = builder.labels;
        scoringFunction = builder.scoringFunction;
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

    public void addInstanceToLabel(DataInstance<Vector> instance, Label label) {
        addInstanceToLabel(new InstanceToLabel(instance, label));
    }

    public void addInstanceToLabel(InstanceToLabel instanceToLabel) {
        instanceToLabel.setInformationGainHeuristicValue(
                scoringFunction.computeInformationGainHeuristicValue(this, instanceToLabel)
        );
        instancesToLabel.add(instanceToLabel);
    }

    public void addInstancesToLabel(List<InstanceToLabel> instancesToLabel) {
        for (InstanceToLabel instanceToLabel : instancesToLabel)
            addInstanceToLabel(instanceToLabel);
    }

    public void removeInstanceToLabel(DataInstance<Vector> instance, Label label) {
        removeInstanceToLabel(new InstanceToLabel(instance, label));
    }

    public void removeInstanceToLabel(InstanceToLabel instanceToLabel) {
        instancesToLabel.remove(instanceToLabel);
    }

    public void removeInstancesToLabel(List<InstanceToLabel> instancesToLabel) {
        for (InstanceToLabel instanceToLabel : instancesToLabel)
            removeInstanceToLabel(instanceToLabel);
    }

    public InstanceToLabel pickInstanceToLabel() {
        if (instancesToLabel.size() > 0)
            return instancesToLabel.first();
        else
            return null;
    }

    public List<InstanceToLabel> pickInstancesToLabel(int numberOfInstancesToPick) {
        List<InstanceToLabel> instances = new ArrayList<>();
        for (InstanceToLabel instanceToLabel : instancesToLabel) {
            instances.add(instanceToLabel);
            if (instances.size() == numberOfInstancesToPick)
                break;
        }
        return instances;
    }

    public void labelInstance(InstanceToLabel instance, Boolean label) {
        if (!dataSet.containsKey(instance.instance))
            dataSet.put(instance.instance, new HashMap<>());
        dataSet.get(instance.instance).put(instance.label, label);
        instancesToLabel.remove(instance);
    }

    public void labelInstances(Map<InstanceToLabel, Boolean> instancesToLabel) {
        for (Map.Entry<InstanceToLabel, Boolean> instance : instancesToLabel.entrySet())
            labelInstance(instance.getKey(), instance.getValue());
    }

    public class InstanceToLabel {
        private final DataInstance<Vector> instance;
        private final Label label;

        private Double informationGainHeuristicValue;

        public InstanceToLabel(DataInstance<Vector> instance, Label label) {
            this.instance = instance;
            this.label = label;
        }

        public DataInstance<Vector> getInstance() {
            return instance;
        }

        public Label getLabel() {
            return label;
        }

        public double getProbability() {
            return probabilityFunction.apply(this);
        }

        public InstanceToLabel setInformationGainHeuristicValue(Double informationGainHeuristicValue) {
            this.informationGainHeuristicValue = informationGainHeuristicValue;
            return this;
        }

        public double getInformationGainHeuristicValue() {
            return informationGainHeuristicValue;
        }

        @Override
        public boolean equals(Object other) {
            if (this == other)
                return true;
            if (other == null || getClass() != other.getClass())
                return false;

            InstanceToLabel that = (InstanceToLabel) other;

            return Objects.equal(label, that.label) && Objects.equal(instance.name(), that.instance.name());
        }

        @Override
        public int hashCode() {
            return Objects.hashCode(label, instance.name());
        }
    }
}
