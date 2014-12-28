package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface DataInstanceWithFeatures<T extends Vector> {
    public String name();
    public T features();
    public DataInstanceBase<T> toDataInstanceBase();
}
