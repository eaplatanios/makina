package org.platanios.learn.classification;

import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.LabeledDataInstance;
import org.platanios.math.matrix.Vector;

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
