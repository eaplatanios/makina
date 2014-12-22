package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class PredictedDataInstance<T extends Vector, S> extends PredictedDataInstanceBase<T, S> {
    protected final T features;

    protected static abstract class AbstractBuilder<T extends Vector, S, B extends AbstractBuilder<T, S, B>>
            extends PredictedDataInstanceBase.AbstractBuilder<T, S, B> {
        protected T features = null;

        protected AbstractBuilder(String name, T features) {
            super(name);
            this.features = features;
        }

        protected AbstractBuilder(PredictedDataInstance<T, S> dataInstance) {
            super(dataInstance);
            this.features = dataInstance.features;
        }

        public B features(T features) {
            this.features = features;
            return self();
        }

        public PredictedDataInstance<T, S> build() {
            return new PredictedDataInstance<>(this);
        }
    }

    public static class Builder<T extends Vector, S> extends AbstractBuilder<T, S, Builder<T, S>> {
        public Builder(String name, T features) {
            super(name, features);
        }

        public Builder(PredictedDataInstance<T, S> dataInstance) {
            super(dataInstance);
        }

        @Override
        protected Builder<T, S> self() {
            return this;
        }
    }

    protected PredictedDataInstance(AbstractBuilder<T, S, ?> builder) {
        super(builder);
        this.features = builder.features;
    }

    protected PredictedDataInstance(String name, T features, S label, Object source, double probability) {
        super(name, label, source, probability);
        this.features = features;
    }

    @Override
    public T features() {
        return features;
    }

    @Override
    public PredictedDataInstanceBase<T, S> toDataInstanceBase() {
        return new PredictedDataInstanceBase<>(name, label, source, probability);
    }
}
