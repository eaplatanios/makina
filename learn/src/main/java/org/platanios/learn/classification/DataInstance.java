package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataInstance<T extends Vector, S> {
    private final String name;
    private final T features;
    private final S label;
    private final double probability;
    private final Object source;

    public static class Builder<T extends Vector, S> {
        private final T features;

        private String name;
        private S label = null;
        private double probability = 1;
        private Object source = null;

        public Builder(T features) {
            this.features = features;
        }

        public Builder(DataInstance<T, S> dataInstance) {
            this.name = dataInstance.name;
            this.features = dataInstance.features;
            this.label = dataInstance.label;
            this.probability = dataInstance.probability;
            this.source = dataInstance.source;
        }

        public Builder<T, S> name(String name) {
            this.name = name;
            return this;
        }

        public Builder<T, S> label(S label) {
            this.label = label;
            return this;
        }

        public Builder<T, S> probability(double probability) {
            this.probability = probability;
            return this;
        }

        public Builder<T, S> source(Object source) {
            this.source = source;
            return this;
        }

        public DataInstance<T, S> build() {
            return new DataInstance<>(this);
        }
    }

    private DataInstance(Builder<T, S> builder) {
        this.name = builder.name;
        this.features = builder.features;
        this.label = builder.label;
        this.probability = builder.probability;
        this.source = builder.source;
    }

    public String name() {
        return name;
    }

    public T features() {
        return features;
    }

    public S label() {
        return label;
    }

    public double probability() {
        return probability;
    }

    public Object source() {
        return source;
    }
}
