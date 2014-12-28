package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface PredictedDataInstanceWithFeatures<T extends Vector, S> extends LabeledDataInstanceWithFeatures<T, S> {
    public double probability();
    @Override
    public PredictedDataInstanceBase<T, S> toDataInstanceBase();
}
