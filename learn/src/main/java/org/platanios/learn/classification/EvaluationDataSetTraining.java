package org.platanios.learn.classification;

import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.LabeledDataInstance;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.math.matrix.Vector;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class EvaluationDataSetTraining<T extends Vector, S> extends Training<T, S> {
    protected final DataSet<? extends LabeledDataInstance<T, S>> evaluationDataSet;
    protected final List<S> evaluationDataSetLabels;

    protected static abstract class AbstractBuilder<B extends AbstractBuilder<B, T, S>, T extends Vector, S>
            extends Training.AbstractBuilder<B, T, S> {
        protected DataSet<? extends LabeledDataInstance<T, S>> evaluationDataSet;
        protected List<S> evaluationDataSetLabels = new ArrayList<>();

        private AbstractBuilder(TrainableClassifier.Builder<T, S> classifierBuilder,
                                DataSet<? extends LabeledDataInstance<T, S>> trainingDataSet,
                                DataSet<? extends LabeledDataInstance<T, S>> evaluationDataSet) {
            super(classifierBuilder, trainingDataSet);

            this.evaluationDataSet = evaluationDataSet;
            for (LabeledDataInstance<T, S> evaluationDataInstance : this.evaluationDataSet)
                evaluationDataSetLabels.add(evaluationDataInstance.label());
        }

        public EvaluationDataSetTraining<T, S> build() {
            return new EvaluationDataSetTraining<>(this);
        }
    }

    public static class Builder<T extends Vector, S> extends AbstractBuilder<Builder<T, S>, T, S> {
        public Builder(TrainableClassifier.Builder<T, S> classifierBuilder,
                       DataSet<? extends LabeledDataInstance<T, S>> trainingDataSet,
                       DataSet<? extends LabeledDataInstance<T, S>> evaluationDataSet) {
            super(classifierBuilder, trainingDataSet, evaluationDataSet);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder<T, S> self() {
            return this;
        }
    }

    private EvaluationDataSetTraining(AbstractBuilder<?, T, S> builder) {
        super(builder);

        evaluationDataSet = builder.evaluationDataSet;
        evaluationDataSetLabels = builder.evaluationDataSetLabels;
    }

    @Override
    protected double trainAndEvaluateClassifier(TrainableClassifier<T, S> classifier) {
        classifier.train(labeledDataSet);
        List<S> predictedLabels = new ArrayList<>();
        for (PredictedDataInstance<T, S> predictedDataInstance : classifier.predict(evaluationDataSet))
            predictedLabels.add(predictedDataInstance.label());
        int[] dataSetIndexes = new int[evaluationDataSet.size()];
        for (int index = 0; index < evaluationDataSet.size(); index++)
            dataSetIndexes[index] = index;
        return computeLoss(predictedLabels, evaluationDataSetLabels, dataSetIndexes);
    }

    @Override
    protected boolean needsTrainingAfterSearch() {
        return false;
    }
}
