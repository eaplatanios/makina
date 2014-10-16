package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface TrainableClassifier<T extends Vector, S> extends Classifier<T, S> {
    public void train();
}
