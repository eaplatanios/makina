package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataInstance<T extends Vector> extends DataInstanceBase<T> {
    protected final T features;

    protected static abstract class AbstractBuilder<T extends Vector, B extends AbstractBuilder<T, B>>
            extends DataInstanceBase.AbstractBuilder<T, B> {
        protected T features = null;

        protected AbstractBuilder(String name, T features) {
            super(name);
            this.features = features;
        }

        protected AbstractBuilder(DataInstance<T> dataInstance) {
            super(dataInstance);
            this.features = dataInstance.features;
        }

        public B features(T features) {
            this.features = features;
            return self();
        }

        public DataInstance<T> build() {
            return new DataInstance<>(this);
        }
    }

    public static class Builder<T extends Vector> extends AbstractBuilder<T, Builder<T>> {
        public Builder(String name, T features) {
            super(name, features);
        }

        public Builder(DataInstance<T> dataInstance) {
            super(dataInstance);
        }

        @Override
        protected Builder<T> self() {
            return this;
        }
    }

    protected DataInstance(AbstractBuilder<T, ?> builder) {
        super(builder);
        this.features = builder.features;
    }

    protected DataInstance(String name, T features) {
        super(name);
        this.features = features;
    }

    @Override
    public T features() {
        return features;
    }

    @Override
    public DataInstanceBase<T> toDataInstanceBase() {
        return new DataInstanceBase<>(name);
    }
}
