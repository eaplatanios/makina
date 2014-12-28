package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface LabeledDataInstanceWithFeatures<T extends Vector, S> extends DataInstanceWithFeatures<T> {
    public S label();
    public Object source();
    @Override
    public LabeledDataInstanceBase<T, S> toDataInstanceBase();
}
