package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LabeledDataInstanceBase<T extends Vector, S> extends DataInstanceBase<T> {
    protected S label;
    protected Object source;

    protected static abstract class AbstractBuilder<T extends Vector, S, B extends AbstractBuilder<T, S, B>>
            extends DataInstanceBase.AbstractBuilder<T, B> {
        protected S label = null;
        protected Object source = null;

        protected AbstractBuilder(String name) {
            super(name);
        }

        protected AbstractBuilder(LabeledDataInstanceBase<T, S> dataInstance) {
            super(dataInstance);
            this.label = dataInstance.label;
            this.source = dataInstance.source;
        }

        public B label(S label) {
            this.label = label;
            return self();
        }

        public B source(Object source) {
            this.source = source;
            return self();
        }

        public LabeledDataInstanceBase<T, S> build() {
            return new LabeledDataInstanceBase<>(this);
        }
    }

    public static class Builder<T extends Vector, S> extends AbstractBuilder<T, S, Builder<T, S>> {
        public Builder(String name) {
            super(name);
        }

        public Builder(LabeledDataInstanceBase<T, S> dataInstance) {
            super(dataInstance);
        }

        @Override
        protected Builder<T, S> self() {
            return this;
        }
    }

    protected LabeledDataInstanceBase(AbstractBuilder<T, S, ?> builder) {
        super(builder);
        this.label = builder.label;
        this.source = builder.source;
    }

    public LabeledDataInstanceBase(String name, S label, Object source) {
        super(name);
        this.label = label;
        this.source = source;
    }

    @Override
    public S label() {
        return label;
    }

    public LabeledDataInstanceBase<T, S> label(S label) {
        this.label = label;
        return this;
    }

    @Override
    public Object source() {
        return source;
    }

    public LabeledDataInstanceBase<T, S> source(Object source) {
        this.source = source;
        return this;
    }

    @Override
    public DataInstanceBase<T> toDataInstance(T features) {
        return new LabeledDataInstance<>(name, features, label, source);
    }

    @Override
    public LabeledDataInstanceBase<T, S> toDataInstanceBase() {
        return this;
    }
}
