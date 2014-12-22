package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataInstanceBase<T extends Vector> {
    protected final String name;

    protected static abstract class AbstractBuilder<T extends Vector, B extends AbstractBuilder<T, B>> {
        protected abstract B self();

        protected String name = null;

        protected AbstractBuilder(String name) {
            this.name = name;
        }

        protected AbstractBuilder(DataInstanceBase dataInstance) {
            this.name = dataInstance.name;
        }

        public B name(String name) {
            this.name = name;
            return self();
        }

        public DataInstanceBase<T> build() {
            return new DataInstanceBase<>(this);
        }
    }

    public static class Builder<T extends Vector> extends AbstractBuilder<T, Builder<T>> {
        public Builder(String name) {
            super(name);
        }

        public Builder(DataInstanceBase dataInstance) {
            super(dataInstance);
        }

        @Override
        protected Builder<T> self() {
            return this;
        }
    }

    protected DataInstanceBase(AbstractBuilder<T, ?> builder) {
        this.name = builder.name;
    }

    protected DataInstanceBase(String name) {
        this.name = name;
    }

    protected DataInstanceBase(DataInstanceBase dataInstance) {
        this.name = dataInstance.name;
    }

    public String name() {
        return name;
    }

    public T features() {
        return null;
    }

    public Object label() {
        return null;
    }

    public Object source() {
        return null;
    }

    public double probability() {
        return 1;
    }

    public DataInstanceBase<T> toDataInstance(T features) {
        return new DataInstance<>(name, features);
    }

    public DataInstanceBase<T> toDataInstanceBase() {
        return this;
    }
}
