package org.platanios.learn.classification.active;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.LabeledDataInstance;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * TODO: Change the code so that we have one data set over everything with a List<Label> as its label type, or something
 * like that.
 *
 * @author Emmanouil Antonios Platanios
 */
public class Learning<V extends Vector> {
    private static final Logger logger = LogManager.getLogger("Classification / Active Learner");

    protected final Set<Label> labels;
    protected final ActiveLearningMethod activeLearningMethod;

    protected Map<Label, DataSet<LabeledDataInstance<V, Double>>> labeledDataSet;
    protected Map<Label, DataSet<PredictedDataInstance<V, Double>>> unlabeledDataSet;

    protected static abstract class AbstractBuilder<V extends Vector, T extends AbstractBuilder<V, T>> {
        /** A self-reference to this builder class. This is basically part of a small "hack" so that we can have
         * inheritable builder classes. */
        protected abstract T self();

        protected final Map<Label, DataSet<LabeledDataInstance<V, Double>>> labeledDataSet;
        protected final Map<Label, DataSet<PredictedDataInstance<V, Double>>> unlabeledDataSet;

        protected Set<Label> labels = new HashSet<>();
        protected ActiveLearningMethod activeLearningMethod = ActiveLearningMethod.UNCERTAINTY_HEURISTIC;

        protected AbstractBuilder(Map<Label, DataSet<LabeledDataInstance<V, Double>>> labeledDataSet,
                                  Map<Label, DataSet<PredictedDataInstance<V, Double>>> unlabeledDataSet) {
            this.labeledDataSet = labeledDataSet;
            this.unlabeledDataSet = unlabeledDataSet;
        }

        public T addLabel(Label label) {
            labels.add(label);
            return self();
        }

        public T activeLearningMethod(ActiveLearningMethod activeLearningMethod) {
            this.activeLearningMethod = activeLearningMethod;
            return self();
        }

        public Learning<V> build() {
            return new Learning<>(this);
        }
    }

    /**
     * The builder class for this abstract class. This is basically part of a small "hack" so that we can have
     * inheritable builder classes.
     */
    public static class Builder<V extends Vector> extends AbstractBuilder<V, Builder<V>> {
        public Builder(Map<Label, DataSet<LabeledDataInstance<V, Double>>> labeledDataSet,
                       Map<Label, DataSet<PredictedDataInstance<V, Double>>> unlabeledDataSet) {
            super(labeledDataSet, unlabeledDataSet);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder<V> self() {
            return this;
        }
    }

    protected Learning(AbstractBuilder<V, ?> builder) {
        labels = builder.labels;
        activeLearningMethod = builder.activeLearningMethod;
        labeledDataSet = builder.labeledDataSet;
        unlabeledDataSet = builder.unlabeledDataSet;
    }

    public Map<Label, DataSet<LabeledDataInstance<V, Double>>> getLabeledDataSet() {
        return labeledDataSet;
    }

    public DataSet<LabeledDataInstance<V, Double>> getLabeledDataSet(Label label) {
        return labeledDataSet.get(label);
    }

    public Map<Label, DataSet<PredictedDataInstance<V, Double>>> getUnlabeledDataSet() {
        return unlabeledDataSet;
    }

    public DataSet<PredictedDataInstance<V, Double>> getUnlabeledDataSet(Label label) {
        return unlabeledDataSet.get(label);
    }

    public int getNumberOfLabeledInstances(Label label) {
        return labeledDataSet.get(label).size();
    }

    public int getNumberOfLabeledInstances() {
        int numberOfInstances = 0;
        for (Label label : labels)
            numberOfInstances += getNumberOfLabeledInstances(label);
        return numberOfInstances;
    }

    public int getNumberOfUnlabeledInstances(Label label) {
        return unlabeledDataSet.get(label).size();
    }

    public int getNumberOfUnlabeledInstances() {
        int numberOfInstances = 0;
        for (Label label : labels)
            numberOfInstances += getNumberOfUnlabeledInstances(label);
        return numberOfInstances;
    }

    public InstanceToLabel<V> pickInstanceToLabel(Map<Label, DataSet<PredictedDataInstance<V, Double>>> dataSets) {
        return activeLearningMethod.pickInstanceToLabel(this, dataSets);
    }

    public List<InstanceToLabel<V>> pickInstancesToLabel(Map<Label, DataSet<PredictedDataInstance<V, Double>>> dataSets,
                                                         int numberOfInstancesToPick) {
        return activeLearningMethod.pickInstancesToLabel(this, dataSets, numberOfInstancesToPick);
    }

    public void labelInstance(InstanceToLabel<V> instance, Double newLabel) {
        Label label = instance.getLabel();
        PredictedDataInstance<V, Double> dataInstance = unlabeledDataSet.get(label).remove(instance.getInstance());
        dataInstance.label(newLabel);
        labeledDataSet.get(label).add(dataInstance);
    }

    public void labelInstances(Map<InstanceToLabel<V>, Double> instancesToLabel) {
        for (Map.Entry<InstanceToLabel<V>, Double> instance : instancesToLabel.entrySet()) {
            Label label = instance.getKey().getLabel();
            PredictedDataInstance<V, Double> dataInstance =
                    unlabeledDataSet.get(label).remove(instance.getKey().getInstance());
            dataInstance.label(instance.getValue());
            labeledDataSet.get(label).add(dataInstance);
        }
    }

    public static class InstanceToLabel<V extends Vector> {
        private final Label label;
        private final PredictedDataInstance<V, Double> instance;

        public InstanceToLabel(Label label, PredictedDataInstance<V, Double> instance) {
            this.label = label;
            this.instance = instance;
        }

        public Label getLabel() {
            return label;
        }

        public PredictedDataInstance<V, Double> getInstance() {
            return instance;
        }
    }
}
