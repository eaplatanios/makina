package makina.learn.classification.reflection;

import makina.learn.classification.Label;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class Integrator {
    protected Data<Data.PredictedInstance> data;

    protected ErrorRates errorRates;
    protected Data<Data.PredictedInstance> integratedData;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>> {
        protected abstract T self();

        protected final Data<Data.PredictedInstance> data;

        protected AbstractBuilder(Data<Data.PredictedInstance> data) {
            this.data = data;
        }

        protected AbstractBuilder(String dataFilename) {
            List<Data.PredictedInstance> predictedInstances = new ArrayList<>();
            if(!loadPredictedInstances(dataFilename, predictedInstances))
                throw new RuntimeException("The integrator data could not be loaded from the provided file.");
            data = new Data<>(predictedInstances);
        }
    }

    protected static class Builder extends AbstractBuilder<Builder> {
        protected Builder(Data<Data.PredictedInstance> data) {
            super(data);
        }

        protected Builder(String dataFilename) {
            super(dataFilename);
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

    public ErrorRates errorRates() {
        return errorRates(false);
    }

    public Data<Data.PredictedInstance> integratedData() {
        return integratedData(false);
    }

    public abstract ErrorRates errorRates(boolean forceComputation);
    public abstract Data<Data.PredictedInstance> integratedData(boolean forceComputation);

    public boolean saveData(String filename) {
        return data != null && savePredictedInstances(filename, data.instances);
    }

    public boolean loadData(String filename) {
        List<Data.PredictedInstance> predictedInstances = new ArrayList<>();
        boolean resultStatus = loadPredictedInstances(filename, predictedInstances);
        data = new Data<>(predictedInstances);
        return resultStatus;
    }

    public boolean saveErrorRates(String filename) {
        if (errorRates == null)
            return false;
        try {
            IntegratorProtos.ErrorRates.Builder builder = IntegratorProtos.ErrorRates.newBuilder();
            errorRates.iterator()
                    .forEachRemaining(errorRate -> builder.addErrorRate(
                            IntegratorProtos.ErrorRate.newBuilder()
                                    .setLabel(errorRate.label().name())
                                    .setFunctionId(errorRate.functionId)
                                    .setValue(errorRate.errorRate)
                    ));
            FileOutputStream fileOutputStream = new FileOutputStream(filename);
            builder.build().writeTo(fileOutputStream);
            fileOutputStream.close();
            return true;
        } catch (IOException exception) {
            return false;
        }
    }

    public boolean loadErrorRates(String filename) {
        try {
            errorRates = new ErrorRates(
                    IntegratorProtos.ErrorRates.parseFrom(new FileInputStream(filename))
                            .getErrorRateList().stream()
                            .map(errorRate -> new ErrorRates.Instance(
                                    new Label(errorRate.getLabel()),
                                    errorRate.getFunctionId(),
                                    errorRate.getValue()
                            ))
                            .collect(Collectors.toList())
            );
            return true;
        } catch (IOException exception) {
            return false;
        }
    }

    public boolean saveIntegratedData(String filename) {
        return integratedData != null && savePredictedInstances(filename, integratedData.instances);
    }

    public boolean loadIntegratedData(String filename) {
        List<Data.PredictedInstance> predictedInstances = new ArrayList<>();
        boolean resultStatus = loadPredictedInstances(filename, predictedInstances);
        integratedData = new Data<>(predictedInstances);
        return resultStatus;
    }

    public static boolean savePredictedInstances(String filename, List<Data.PredictedInstance> predictedInstances) {
        try {
            IntegratorProtos.PredictedInstances.Builder builder = IntegratorProtos.PredictedInstances.newBuilder();
            predictedInstances.iterator()
                    .forEachRemaining(predictedInstance -> builder.addPredictedInstance(
                            IntegratorProtos.PredictedInstance.newBuilder()
                                    .setId(predictedInstance.id())
                                    .setLabel(predictedInstance.label().name())
                                    .setFunctionId(predictedInstance.functionId)
                                    .setValue(predictedInstance.value)
                    ));
            FileOutputStream fileOutputStream = new FileOutputStream(filename);
            builder.build().writeTo(fileOutputStream);
            fileOutputStream.close();
            return true;
        } catch (IOException exception) {
            return false;
        }
    }

    public static boolean loadPredictedInstances(String filename, List<Data.PredictedInstance> predictedInstances) {
        try {
            predictedInstances.addAll(IntegratorProtos.PredictedInstances.parseFrom(new FileInputStream(filename))
                    .getPredictedInstanceList().stream()
                    .map(predictedInstance -> new Data.PredictedInstance(
                            predictedInstance.getId(),
                            new Label(predictedInstance.getLabel()),
                            predictedInstance.getFunctionId(),
                            predictedInstance.getValue()
                    ))
                    .collect(Collectors.toList()));
            return true;
        } catch (IOException exception) {
            return false;
        }
    }

    public static boolean saveObservedInstances(String filename, List<Data.ObservedInstance> observedInstances) {
        try {
            IntegratorProtos.ObservedInstances.Builder builder = IntegratorProtos.ObservedInstances.newBuilder();
            observedInstances.iterator()
                    .forEachRemaining(observedInstance -> builder.addObservedInstance(
                            IntegratorProtos.ObservedInstance.newBuilder()
                                    .setId(observedInstance.id())
                                    .setLabel(observedInstance.label().name())
                                    .setValue(observedInstance.value)
                    ));
            FileOutputStream fileOutputStream = new FileOutputStream(filename);
            builder.build().writeTo(fileOutputStream);
            fileOutputStream.close();
            return true;
        } catch (IOException exception) {
            return false;
        }
    }

    public static boolean loadObservedInstances(String filename, List<Data.ObservedInstance> observedInstances) {
        try {
            observedInstances.addAll(IntegratorProtos.ObservedInstances.parseFrom(new FileInputStream(filename))
                    .getObservedInstanceList().stream()
                    .map(observedInstance -> new Data.ObservedInstance(
                            observedInstance.getId(),
                            new Label(observedInstance.getLabel()),
                            observedInstance.getValue()
                    ))
                    .collect(Collectors.toList()));
            return true;
        } catch (IOException exception) {
            return false;
        }
    }

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
            private final int id;
            private final Label label;

            public Instance(int id, Label label) {
                this.id = id;
                this.label = label;
            }

            public int id() {
                return id;
            }

            public Label label() {
                return label;
            }
        }

        public static class ObservedInstance extends Instance {
            private final boolean value;

            public ObservedInstance(int id, Label label, boolean value) {
                super(id, label);
                this.value = value;
            }

            public boolean value() {
                return value;
            }
        }

        public static class PredictedInstance extends Instance {
            private final int functionId;
            private final double value;

            public PredictedInstance(int id, Label label, double value) {
                super(id, label);
                this.functionId = -1;
                this.value = value;
            }

            public PredictedInstance(int id, Label label, int functionId, double value) {
                super(id, label);
                this.functionId = functionId;
                this.value = value;
            }

            public int functionId() {
                return functionId;
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
            private final int functionId;
            private final double errorRate;

            public Instance(Label label, int functionId, double errorRate) {
                this.label = label;
                this.functionId = functionId;
                this.errorRate = errorRate;
            }

            public Label label() {
                return label;
            }

            public int functionId() {
                return functionId;
            }

            public double errorRate() {
                return errorRate;
            }
        }
    }
}
