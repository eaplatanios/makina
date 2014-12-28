package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LabeledDataInstance<T extends Vector, S>
        extends LabeledDataInstanceBase<T, S>
        implements LabeledDataInstanceWithFeatures<T, S> {
    protected final T features;

    public LabeledDataInstance(String name, T features, S label, Object source) {
        super(name, label, source);
        this.features = features;
    }

    @Override
    public T features() {
        return features;
    }

    @Override
    public LabeledDataInstanceBase<T, S> toDataInstanceBase() {
        return new LabeledDataInstanceBase<>(name, label, source);
    }
}
