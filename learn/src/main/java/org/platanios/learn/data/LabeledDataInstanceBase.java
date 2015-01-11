package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
class LabeledDataInstanceBase<T extends Vector, S> extends DataInstanceBase<T> {
    protected S label;
    protected Object source;

    public LabeledDataInstanceBase(String name, S label, Object source) {
        super(name);
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
    public LabeledDataInstance<T, S> toDataInstance(T features) {
        return new LabeledDataInstance<>(name, features, label, source);
    }

    @Override
    public MultiViewDataInstance<T> toMultiViewDataInstance(List<T> features) {
        return new MultiViewLabeledDataInstance<>(name, features, label, source);
    }
}
