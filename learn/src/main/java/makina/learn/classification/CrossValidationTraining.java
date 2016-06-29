package makina.learn.classification;

import makina.learn.data.DataSet;
import makina.learn.data.LabeledDataInstance;
import makina.learn.data.PredictedDataInstance;
import makina.math.matrix.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * @author Emmanouil Antonios Platanios
 */
public class CrossValidationTraining<T extends Vector, S> extends Training<T, S> {
    private final int numberOfFolds;
    private final Function<Integer, DataSetPartitioning<T, S>> dataSetPartitioningFunction;

    protected DataSet<? extends LabeledDataInstance<T, S>> trainingDataSet;
    protected DataSet<? extends LabeledDataInstance<T, S>> evaluationDataSet;

    protected static abstract class AbstractBuilder<B extends AbstractBuilder<B, T, S>, T extends Vector, S>
            extends Training.AbstractBuilder<B, T, S> {
        protected int numberOfFolds = 10;
        protected Function<Integer, DataSetPartitioning<T, S>> dataSetPartitioningFunction =
                foldNumber -> {
                    int foldSize = Math.floorDiv(labeledDataSet.size(), numberOfFolds);
                    int[] evaluationDataSetIndexes = new int[foldSize];
                    for (int index = 0; index < foldSize; index++)
                        evaluationDataSetIndexes[index] = index + foldNumber * foldSize;
                    return new DataSetPartitioning<>(
                            labeledDataSet.subSetComplement(foldNumber * foldSize, (foldNumber + 1) * foldSize),
                            labeledDataSet.subSet(foldNumber * foldSize, (foldNumber + 1) * foldSize),
                            evaluationDataSetIndexes
                    );
                };

        private AbstractBuilder(TrainableClassifier.Builder<T, S> classifierBuilder,
                                DataSet<? extends LabeledDataInstance<T, S>> trainingDataSet) {
            super(classifierBuilder, trainingDataSet);
        }

        public B numberOfFolds(int numberOfFolds) {
            this.numberOfFolds = numberOfFolds;
            return self();
        }

        public B dataSetPartitioningFunction(Function<Integer, DataSetPartitioning<T, S>> dataSetPartitioningFunction) {
            this.dataSetPartitioningFunction = dataSetPartitioningFunction;
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
        dataSetPartitioningFunction = builder.dataSetPartitioningFunction;
    }

    @Override
    protected double trainAndEvaluateClassifier(TrainableClassifier<T, S> classifier) {
        double averageLoss = 0;
        for (int fold = 0; fold < numberOfFolds; fold++) {
            DataSetPartitioning<T, S> dataSetPartitioning = dataSetPartitioningFunction.apply(fold);
            trainingDataSet = dataSetPartitioning.getTrainingDataSet();
            evaluationDataSet = dataSetPartitioning.getEvaluationDataSet();
            List<S> evaluationDataSetLabels = new ArrayList<>();
            for (LabeledDataInstance<T, S> evaluationDataInstance : evaluationDataSet)
                evaluationDataSetLabels.add(evaluationDataInstance.label());
            classifier.train(trainingDataSet);
            List<S> predictedLabels = new ArrayList<>();
            for (PredictedDataInstance<T, S> predictedDataInstance : classifier.predict(evaluationDataSet))
                predictedLabels.add(predictedDataInstance.label());
            averageLoss += computeLoss(predictedLabels,
                                       evaluationDataSetLabels,
                                       dataSetPartitioning.getEvaluationDataSetIndexes());
        }
        return averageLoss / numberOfFolds;
    }

    @Override
    protected boolean needsTrainingAfterSearch() {
        return true;
    }

    public static class DataSetPartitioning<T extends Vector, S> {
        private final DataSet<? extends LabeledDataInstance<T, S>> trainingDataSet;
        private final DataSet<? extends LabeledDataInstance<T, S>> evaluationDataSet;
        private final int[] evaluationDataSetIndexes;

        public DataSetPartitioning(DataSet<? extends LabeledDataInstance<T, S>> trainingDataSet,
                                   DataSet<? extends LabeledDataInstance<T, S>> evaluationDataSet,
                                   int[] evaluationDataSetIndexes) {
            this.trainingDataSet = trainingDataSet;
            this.evaluationDataSet = evaluationDataSet;
            this.evaluationDataSetIndexes = evaluationDataSetIndexes;
        }

        public DataSet<? extends LabeledDataInstance<T, S>> getTrainingDataSet() {
            return trainingDataSet;
        }

        public DataSet<? extends LabeledDataInstance<T, S>> getEvaluationDataSet() {
            return evaluationDataSet;
        }

        public int[] getEvaluationDataSetIndexes() {
            return evaluationDataSetIndexes;
        }
    }
}
