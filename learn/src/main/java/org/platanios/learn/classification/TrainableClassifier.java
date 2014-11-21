package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface TrainableClassifier<T extends Vector, S> extends Classifier<T, S> {
    public boolean train(List<DataInstance<T, S>> trainingData);
}
