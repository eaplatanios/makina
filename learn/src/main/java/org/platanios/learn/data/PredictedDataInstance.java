package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class PredictedDataInstance<T extends Vector, S>
        extends PredictedDataInstanceBase<T, S>
        implements PredictedDataInstanceWithFeatures<T, S> {
    protected final T features;

    public PredictedDataInstance(String name, T features, S label, Object source, double probability) {
        super(name, label, source, probability);
        this.features = features;
    }

    @Override
    public T features() {
        return features;
    }

    @Override
    public PredictedDataInstanceBase<T, S> toDataInstanceBase() {
        return new PredictedDataInstanceBase<>(name, label, source, probability);
    }
}
