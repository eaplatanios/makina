package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LabeledDataInstance<T extends Vector, S> extends DataInstance<T> {
    protected S label;
    protected Object source;

    public LabeledDataInstance(String name, S label, Object source) {
        super(name);
        this.label = label;
        this.source = source;
    }

    public LabeledDataInstance(String name, T features, S label, Object source) {
        super(name, features);
        this.label = label;
        this.source = source;
    }

    public S label() {
        return label;
    }

    public void label(S label) {
        this.label = label;
    }

    public Object source() {
        return source;
    }

    public void source(Object source) {
        this.source = source;
    }

    @Override
    protected LabeledDataInstanceBase<T, S> toDataInstanceBase() {
        return new LabeledDataInstanceBase<>(name, label, source);
    }
}
