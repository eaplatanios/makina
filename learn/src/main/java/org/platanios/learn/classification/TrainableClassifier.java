package org.platanios.learn.classification;

import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.LabeledDataInstance;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface TrainableClassifier<T extends Vector, S> extends Classifier<T, S> {
    public boolean train(DataSet<? extends LabeledDataInstance<T, S>> trainingDataSet);

    public interface Builder<T extends Vector, S> {
        public Builder<T, S> setParameter(String name, Object value);
        public TrainableClassifier<T, S> build();
    }
}
