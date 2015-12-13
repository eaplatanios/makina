package org.platanios.learn.classification.active;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.classification.TrainableClassifier;
import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.LabeledDataInstance;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;

import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Learning<V extends Vector> {
    private static final Logger logger = LogManager.getLogger("Classification / Active");

    private final Set<Label> labels;
    private final Map<Label, TrainableClassifier<V, Double>> classifiers;
    private final ActiveLearningMethod activeLearningMethod;

    protected static abstract class AbstractBuilder<V extends Vector, T extends AbstractBuilder<V, T>> {
        /** A self-reference to this builder class. This is basically part of a small "hack" so that we can have
         * inheritable builder classes. */
        protected abstract T self();

        protected Set<Label> labels = new HashSet<>();
        protected Map<Label, TrainableClassifier<V, Double>> classifiers = new HashMap<>();
        protected ActiveLearningMethod activeLearningMethod = ActiveLearningMethod.UNCERTAINTY_HEURISTIC;

        protected AbstractBuilder() { }

        public T addLabel(Label label, TrainableClassifier<V, Double> classifier) {
            labels.add(label);
            classifiers.put(label, classifier);
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
        public Builder() {
            super();
        }

        /** {@inheritDoc} */
        @Override
        protected Builder<V> self() {
            return this;
        }
    }

    protected Learning(AbstractBuilder<V, ?> builder) {
        labels = builder.labels;
        classifiers = builder.classifiers;
        activeLearningMethod = builder.activeLearningMethod;
    }

    public void trainClassifier(Label label, DataSet<LabeledDataInstance<V, Double>> dataSet) {
        classifiers.get(label).train(dataSet);
    }

    public void trainClassifiers(Map<Label, DataSet<LabeledDataInstance<V, Double>>> dataSet) {
        for (Label label : labels)
            trainClassifier(label, dataSet.get(label));
    }

    public void makePredictions(Label label, DataSet<PredictedDataInstance<V, Double>> dataSet) {
        classifiers.get(label).predictInPlace(dataSet);
    }

    public void makePredictions(Map<Label, DataSet<PredictedDataInstance<V, Double>>> dataSet) {
        for (Label label : labels)
            makePredictions(label, dataSet.get(label));
    }

    public InstanceToLabel<V> pickInstanceToLabel(Map<Label, DataSet<PredictedDataInstance<V, Double>>> dataSets) {
        return activeLearningMethod.pickInstanceToLabel(this, dataSets);
    }

    public List<InstanceToLabel<V>> pickInstancesToLabel(Map<Label, DataSet<PredictedDataInstance<V, Double>>> dataSets,
                                                         int numberOfInstancesToPick) {
        return activeLearningMethod.pickInstancesToLabel(this, dataSets, numberOfInstancesToPick);
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
