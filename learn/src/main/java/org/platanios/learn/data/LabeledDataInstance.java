package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LabeledDataInstance<T extends Vector, S> extends LabeledDataInstanceBase<T, S> {
    protected final T features;

    protected static abstract class AbstractBuilder<T extends Vector, S, B extends AbstractBuilder<T, S, B>>
            extends LabeledDataInstanceBase.AbstractBuilder<T, S, B> {
        protected T features = null;

        protected AbstractBuilder(String name, T features) {
            super(name);
            this.features = features;
        }

        protected AbstractBuilder(LabeledDataInstance<T, S> dataInstance) {
            super(dataInstance);
            this.features = dataInstance.features;
        }

        public B features(T features) {
            this.features = features;
            return self();
        }

        public LabeledDataInstance<T, S> build() {
            return new LabeledDataInstance<>(this);
        }
    }

    public static class Builder<T extends Vector, S> extends AbstractBuilder<T, S, Builder<T, S>> {
        public Builder(String name, T features) {
            super(name, features);
        }

        public Builder(LabeledDataInstance<T, S> dataInstance) {
            super(dataInstance);
        }

        @Override
        protected Builder<T, S> self() {
            return this;
        }
    }

    protected LabeledDataInstance(AbstractBuilder<T, S, ?> builder) {
        super(builder);
        this.features = builder.features;
    }

    protected LabeledDataInstance(String name, T features, S label, Object source) {
        super(name, label, source);
        this.features = features;
    }

    @Override
    public T features() {
        return features;
    }

    @Override
    public LabeledDataInstanceBase<T, S> toDataInstanceBase() {
        return new LabeledDataInstanceBase<>(name, label, source);
    }
}
