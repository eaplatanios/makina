package org.platanios.learn.classification.active;

import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.LabeledDataInstance;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;

import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ConstrainedLearning<V extends Vector> extends Learning<V> {
    private final ConstraintSet constraints;

    private final Map<String, Map<Label, Boolean>> fixedLabels = new HashMap<>();

    protected static abstract class AbstractBuilder<V extends Vector, T extends AbstractBuilder<V, T>>
            extends Learning.AbstractBuilder<V, T> {
        private Set<Constraint> constraints = new HashSet<>();

        protected AbstractBuilder(Map<Label, DataSet<LabeledDataInstance<V, Double>>> labeledDataSet,
                                  Map<Label, DataSet<PredictedDataInstance<V, Double>>> unlabeledDataSet) {
            super(labeledDataSet, unlabeledDataSet);
        }

        public T addConstraint(Constraint constraint) {
            constraints.add(constraint);
            return self();
        }

        public T addMutualExclusionConstraint(Set<Label> labels) {
            constraints.add(new MutualExclusionConstraint(labels));
            return self();
        }

        public T addMutualExclusionConstraint(Label... labels) {
            constraints.add(new MutualExclusionConstraint(new HashSet<>(Arrays.asList(labels))));
            return self();
        }

        public ConstrainedLearning<V> build() {
            return new ConstrainedLearning<>(this);
        }
    }

    /**
     * The builder class for this abstract class. This is basically part of a small "hack" so that we can have
     * inheritable builder classes.
     */
    public static class Builder<V extends Vector> extends AbstractBuilder<V, Builder<V>> {
        public Builder(Map<Label, DataSet<LabeledDataInstance<V, Double>>> labeledDataSet,
                       Map<Label, DataSet<PredictedDataInstance<V, Double>>> unlabeledDataSet) {
            super(labeledDataSet, unlabeledDataSet);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder<V> self() {
            return this;
        }
    }

    private ConstrainedLearning(AbstractBuilder<V, ?> builder) {
        super(builder);
        constraints = new ConstraintSet(builder.constraints);
        for (Label label : labels)
            for (LabeledDataInstance<V, Double> dataInstance : labeledDataSet.get(label)) {
                if (!fixedLabels.containsKey(dataInstance.name()))
                    fixedLabels.put(dataInstance.name(), new HashMap<>());
                fixedLabels.get(dataInstance.name()).put(label, dataInstance.label() >= 0.5);
            }
        propagateConstraints(labeledDataSet, unlabeledDataSet);
    }

    public Constraint getConstraints() {
        return constraints;
    }

    public Map<String, Map<Label, Boolean>> getFixedLabels() {
        return fixedLabels;
    }

    public Map<Label, Boolean> getFixedLabels(String instanceName) {
        return fixedLabels.getOrDefault(instanceName, new HashMap<>());
    }

    @Override
    public void labelInstance(InstanceToLabel<V> instance, Double newLabel) {
        super.labelInstance(instance, newLabel);
        String instanceName = instance.getInstance().name();
        if (!fixedLabels.containsKey(instanceName))
            fixedLabels.put(instanceName, new HashMap<>());
        fixedLabels.get(instanceName).put(instance.getLabel(), newLabel >= 0.5);
        propagateConstraints(labeledDataSet, unlabeledDataSet);
    }

    @Override
    public void labelInstances(Map<InstanceToLabel<V>, Double> instancesToLabel) {
        super.labelInstances(instancesToLabel);
        for (Map.Entry<InstanceToLabel<V>, Double> instance : instancesToLabel.entrySet()) {
            String instanceName = instance.getKey().getInstance().name();
            if (!fixedLabels.containsKey(instanceName))
                fixedLabels.put(instanceName, new HashMap<>());
            fixedLabels.get(instanceName).put(instance.getKey().getLabel(), instance.getValue() >= 0.5);
        }
        propagateConstraints(labeledDataSet, unlabeledDataSet);
    }

    // TODO: Make this method more efficient.
    private int propagateConstraints(Map<Label, DataSet<LabeledDataInstance<V, Double>>> labeledDataSet,
                                     Map<Label, DataSet<PredictedDataInstance<V, Double>>> unlabeledDataSet) {
        int numberOfLabelsFixed = 0;
        for (Map.Entry<String, Map<Label, Boolean>> fixedLabelsEntry : fixedLabels.entrySet()) {
            if (constraints.propagate(fixedLabelsEntry.getValue()) > 0)
                for (Map.Entry<Label, Boolean> fixedLabel : fixedLabelsEntry.getValue().entrySet()) {
                    for (PredictedDataInstance<V, Double> dataInstance : unlabeledDataSet.get(fixedLabel.getKey())) {
                        if (dataInstance.name().equals(fixedLabelsEntry.getKey())) {
                            unlabeledDataSet.get(fixedLabel.getKey()).remove(dataInstance);
                            dataInstance.label(fixedLabel.getValue() ? 1.0 : 0.0);
                            labeledDataSet.get(fixedLabel.getKey()).add(dataInstance);
                            numberOfLabelsFixed++;
                        }
                    }
                }
        }
        return numberOfLabelsFixed;
    }
}
