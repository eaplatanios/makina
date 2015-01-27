package org.platanios.learn.classification;

import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.LabeledDataInstance;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class CrossValidationTraining<T extends Vector, S> extends Training<T, S> {
    private final int numberOfFolds;
    protected final List<S> fullDataSetLabels;

    protected DataSet<? extends LabeledDataInstance<T, S>> trainingDataSet;
    protected DataSet<? extends LabeledDataInstance<T, S>> evaluationDataSet;

    protected static abstract class AbstractBuilder<B extends AbstractBuilder<B, T, S>, T extends Vector, S>
            extends Training.AbstractBuilder<B, T, S> {
        protected int numberOfFolds = 10;
        protected List<S> fullDataSetLabels = new ArrayList<>();

        private AbstractBuilder(TrainableClassifier.Builder<T, S> classifierBuilder,
                                DataSet<? extends LabeledDataInstance<T, S>> trainingDataSet) {
            super(classifierBuilder, trainingDataSet);

            for (LabeledDataInstance<T, S> labeledDataInstance : labeledDataSet)
                fullDataSetLabels.add(labeledDataInstance.label());
        }

        public B numberOfFolds(int numberOfFolds) {
            this.numberOfFolds = numberOfFolds;
            return self();
        }

        public CrossValidationTraining<T, S> build() {
            return new CrossValidationTraining<>(this);
        }
    }

    public static class Builder<T extends Vector, S> extends AbstractBuilder<Builder<T, S>, T, S> {
        public Builder(TrainableClassifier.Builder<T, S> classifierBuilder,
                       DataSet<? extends LabeledDataInstance<T, S>> trainingDataSet) {
            super(classifierBuilder, trainingDataSet);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder<T, S> self() {
            return this;
        }
    }

    private CrossValidationTraining(AbstractBuilder<?, T, S> builder) {
        super(builder);

        numberOfFolds = builder.numberOfFolds;
        fullDataSetLabels = builder.fullDataSetLabels;
    }

    @Override
    protected double trainAndEvaluateClassifier(TrainableClassifier<T, S> classifier) {
        double averageLoss = 0;
        int foldSize = Math.floorDiv(labeledDataSet.size(), numberOfFolds);
        for (int fold = 0; fold < numberOfFolds; fold++) {
            trainingDataSet = labeledDataSet.subSet(fold * foldSize, (fold + 1) * foldSize);
            evaluationDataSet = labeledDataSet.subSetComplement(fold * foldSize, (fold + 1) * foldSize);
            List<S> evaluationDataSetLabels = new ArrayList<>();
            for (LabeledDataInstance<T, S> evaluationDataInstance : evaluationDataSet)
                evaluationDataSetLabels.add(evaluationDataInstance.label());
            classifier.train(trainingDataSet);
            List<S> predictedLabels = new ArrayList<>();
            for (PredictedDataInstance<T, S> predictedDataInstance : classifier.predict(evaluationDataSet))
                predictedLabels.add(predictedDataInstance.label());
            averageLoss += lossFunction.computeLoss(predictedLabels, evaluationDataSetLabels);
        }
        return averageLoss / numberOfFolds;
    }
}
