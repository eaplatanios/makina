package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataInstance<T extends Vector> extends DataInstanceBase<T> implements DataInstanceWithFeatures<T> {
    protected final T features;

    public DataInstance(String name, T features) {
        super(name);
        this.features = features;
    }

    @Override
    public T features() {
        return features;
    }

    @Override
    public DataInstanceBase<T> toDataInstanceBase() {
        return new DataInstanceBase<>(name);
    }
}
