package org.platanios.learn.classification.reflection;

import org.platanios.learn.classification.Label;

import java.util.Iterator;
import java.util.List;
import java.util.stream.Stream;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class Integrator {
    protected final Data<Data.PredictedInstance> data;

    protected ErrorRates errorRates;
    protected Data<Data.PredictedInstance> integratedData;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>> {
        protected abstract T self();

        protected final Data<Data.PredictedInstance> data;

        protected AbstractBuilder(Data<Data.PredictedInstance> data) {
            this.data = data;
        }
    }

    protected static class Builder extends AbstractBuilder<Builder> {
        protected Builder(Data<Data.PredictedInstance> data) {
            super(data);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    protected Integrator(AbstractBuilder<?> builder) {
        data = builder.data;
        errorRates = null;
        integratedData = null;
    }

    public abstract ErrorRates errorRates();
    public abstract Data<Data.PredictedInstance> integratedData();

    public static class Data<T extends Data.Instance> implements Iterable<T> {
        private final List<T> instances;

        public Data(List<T> instances) {
            this.instances = instances;
        }

        public int size() {
            return instances.size();
        }

        public T get(int index) {
            if (index >= instances.size())
                throw new IllegalArgumentException("The provided instance index is out of bounds.");
            return instances.get(index);
        }

        @Override
        public Iterator<T> iterator() {
            return instances.iterator();
        }

        public Stream<T> stream() {
            return instances.stream();
        }

        public static abstract class Instance {
            private final int instanceId;
            private final Label label;

            public Instance(int instanceId, Label label) {
                this.instanceId = instanceId;
                this.label = label;
            }

            public int instanceId() {
                return instanceId;
            }

            public Label label() {
                return label;
            }
        }

        public static class ObservedInstance extends Instance {
            private final boolean value;

            public ObservedInstance(int instanceID, Label label, boolean value) {
                super(instanceID, label);
                this.value = value;
            }

            public boolean value() {
                return value;
            }
        }

        public static class PredictedInstance extends Instance {
            private final int classifierId;
            private final double value;

            public PredictedInstance(int instanceID, Label label, int classifierId, double value) {
                super(instanceID, label);
                this.classifierId = classifierId;
                this.value = value;
            }

            public int classifierId() {
                return classifierId;
            }

            public double value() {
                return value;
            }
        }
    }

    public static class ErrorRates implements Iterable<ErrorRates.Instance> {
        private final List<Instance> instances;

        public ErrorRates(List<Instance> instances) {
            this.instances = instances;
        }

        public int size() {
            return instances.size();
        }

        public Instance get(int index) {
            if (index >= instances.size())
                throw new IllegalArgumentException("The provided instance index is out of bounds.");
            return instances.get(index);
        }

        @Override
        public Iterator<Instance> iterator() {
            return instances.iterator();
        }

        public Stream<Instance> stream() {
            return instances.stream();
        }

        public static class Instance {
            private final Label label;
            private final int classifierID;
            private final double errorRate;

            public Instance(Label label, int classifierID, double errorRate) {
                this.label = label;
                this.classifierID = classifierID;
                this.errorRate = errorRate;
            }

            public Label label() {
                return label;
            }

            public int classifierID() {
                return classifierID;
            }

            public double errorRate() {
                return errorRate;
            }
        }
    }
}
