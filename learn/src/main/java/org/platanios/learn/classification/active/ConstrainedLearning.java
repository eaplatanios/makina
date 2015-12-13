package org.platanios.learn.classification.active;

import org.platanios.learn.math.matrix.Vector;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ConstrainedLearning<V extends Vector> extends Learning<V> {
    private final ConstraintSet constraints;

    protected static abstract class AbstractBuilder<V extends Vector, T extends AbstractBuilder<V, T>>
            extends Learning.AbstractBuilder<V, T> {
        private Set<Constraint> constraints = new HashSet<>();

        protected AbstractBuilder() { }

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
        public Builder() {
            super();
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
    }
}
