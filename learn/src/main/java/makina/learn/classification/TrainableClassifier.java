package makina.learn.classification;

import makina.learn.data.DataSet;
import makina.learn.data.LabeledDataInstance;
import makina.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface TrainableClassifier<T extends Vector, S> extends Classifier<T, S> {
    boolean train(DataSet<? extends LabeledDataInstance<T, S>> trainingDataSet);

    interface Builder<T extends Vector, S> {
        Builder<T, S> setParameter(String name, Object value);
        TrainableClassifier<T, S> build();
    }
}
