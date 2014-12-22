package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class PredictedDataInstanceBase<T extends Vector, S> extends LabeledDataInstanceBase<T, S> {
    protected final double probability;

    protected static abstract class AbstractBuilder<T extends Vector, S, B extends AbstractBuilder<T, S, B>>
            extends LabeledDataInstanceBase.AbstractBuilder<T, S, B> {
        protected double probability = 1;

        protected AbstractBuilder(String name) {
            super(name);
        }

        protected AbstractBuilder(PredictedDataInstanceBase<T, S> dataInstance) {
            super(dataInstance);
            this.probability = dataInstance.probability;
        }

        public B probability(double probability) {
            this.probability = probability;
            return self();
        }

        public PredictedDataInstanceBase<T, S> build() {
            return new PredictedDataInstanceBase<>(this);
        }
    }

    public static class Builder<T extends Vector, S> extends AbstractBuilder<T, S, Builder<T, S>> {
        public Builder(String name) {
            super(name);
        }

        public Builder(PredictedDataInstanceBase<T, S> dataInstance) {
            super(dataInstance);
        }

        @Override
        protected Builder<T, S> self() {
            return this;
        }
    }

    protected PredictedDataInstanceBase(AbstractBuilder<T, S, ?> builder) {
        super(builder);
        this.probability = builder.probability;
    }

    protected PredictedDataInstanceBase(String name, S label, Object source, double probability) {
        super(name, label, source);
        this.probability = probability;
    }

    @Override
    public double probability() {
        return probability;
    }

    @Override
    public DataInstanceBase<T> toDataInstance(T features) {
        return new PredictedDataInstance<>(name, features, label, source, probability);
    }

    @Override
    public PredictedDataInstanceBase<T, S> toDataInstanceBase() {
        return this;
    }
}
