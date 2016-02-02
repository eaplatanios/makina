package org.platanios.learn.classification.active;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import org.platanios.learn.classification.Label;
import org.platanios.learn.classification.constraint.Constraint;
import org.platanios.learn.classification.constraint.ConstraintSet;
import org.platanios.learn.classification.constraint.MutualExclusionConstraint;
import org.platanios.learn.data.DataInstance;
import org.platanios.learn.math.matrix.Vector;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ConstrainedLearning extends Learning {
    private final Set<Label> leafLabels = new HashSet<>();

    private final ConstraintSet constraintSet;

    private final Table<DataInstance<Vector>, Label, InstanceToLabel> instancesTable = HashBasedTable.create();

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>> extends Learning.AbstractBuilder<T> {
        private Set<Constraint> constraintsSet = new HashSet<>();

        protected AbstractBuilder(Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet,
                                  Function<InstanceToLabel, Double> probabilityFunction) {
            super(dataSet, probabilityFunction);
        }

        public T addConstraint(Constraint constraint) {
            constraintsSet.add(constraint);
            return self();
        }

        public T addConstraints(Set<Constraint> constraints) {
            constraintsSet.addAll(constraints);
            return self();
        }

        public T addMutualExclusionConstraint(Set<Label> labels) {
            constraintsSet.add(new MutualExclusionConstraint(labels));
            return self();
        }

        public T addMutualExclusionConstraint(Label... labels) {
            constraintsSet.add(new MutualExclusionConstraint(new HashSet<>(Arrays.asList(labels))));
            return self();
        }

        public ConstrainedLearning build() {
            return new ConstrainedLearning(this);
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

    private ConstrainedLearning(AbstractBuilder<?> builder) {
        super(builder);
        constraintSet = new ConstraintSet(builder.constraintsSet);
        propagateConstraints(dataSet);
    }

    public Constraint getConstraintSet() {
        return constraintSet;
    }

    public boolean containsInstanceToLabel(DataInstance<Vector> instance, Label label) {
        return instancesTable.contains(instance, label);
    }

    public InstanceToLabel getInstanceToLabel(DataInstance<Vector> instance, Label label) {
        return instancesTable.get(instance, label);
    }

    @Override
    public void addInstanceToLabel(InstanceToLabel instanceToLabel) {
        instanceToLabel.setInformationGainHeuristicValue(
                scoringFunction.computeInformationGainHeuristicValue(this, instanceToLabel)
        );
        instancesToLabel.add(instanceToLabel);
        instancesTable.put(instanceToLabel.getInstance(), instanceToLabel.getLabel(), instanceToLabel);
    }

    @Override
    public void labelInstance(InstanceToLabel instance, Boolean label) {
        super.labelInstance(instance, label);
        propagateInstanceConstraints(instance.getInstance(), dataSet.get(instance.getInstance()));
    }

    public void labelInstanceWithoutPropagation(DataInstance<Vector> instance, Label label, Boolean value) {
        super.labelInstance(instancesTable.get(instance, label), value);
    }

    private int propagateInstanceConstraints(DataInstance<Vector> instance, Map<Label, Boolean> instanceLabels) {
        return constraintSet.propagate(instanceLabels, this, instance);
    }

    private int propagateConstraints(Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet) {
        int numberOfLabelsFixed = 0;
        for (Map.Entry<DataInstance<Vector>, Map<Label, Boolean>> dataSetEntry : dataSet.entrySet())
            numberOfLabelsFixed += propagateInstanceConstraints(dataSetEntry.getKey(), dataSetEntry.getValue());
        return numberOfLabelsFixed;
    }
}
