package org.platanios.learn.classification.active;

import org.platanios.learn.classification.Label;
import org.platanios.learn.classification.constraint.Constraint;
import org.platanios.learn.classification.constraint.ConstraintSet;
import org.platanios.learn.classification.constraint.MutualExclusionConstraint;
import org.platanios.learn.data.DataInstance;
import org.platanios.learn.math.matrix.Vector;

import java.util.*;
import java.util.function.Function;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ConstrainedLearning extends Learning {
    private final ConstraintSet constraintSet;

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

    @Override
    public void labelInstance(InstanceToLabel instance, Boolean label) {
        super.labelInstance(instance, label);
        propagateInstanceConstraints(dataSet.get(instance.getInstance()));
    }

    private int propagateInstanceConstraints(Map<Label, Boolean> instanceLabels) {
        return constraintSet.propagate(instanceLabels);
    }

    private int propagateConstraints(Map<DataInstance<Vector>, Map<Label, Boolean>> dataSet) {
        int numberOfLabelsFixed = 0;
        for (Map<Label, Boolean> instanceLabelsMap : dataSet.values())
            numberOfLabelsFixed += propagateInstanceConstraints(instanceLabelsMap);
        return numberOfLabelsFixed;
    }
}
