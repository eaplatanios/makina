package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface Classifier<T extends Vector, S> {
    public double predict(DataInstance<T, S> dataInstance);
    public double[] predict(DataInstance<Vector, Integer>[] dataInstances);
}
