package org.platanios.learn.data;

import org.platanios.math.matrix.Vector;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class MultiViewLabeledDataInstance<T extends Vector, S> extends MultiViewDataInstance<T> {
    protected S label;
    protected Object source;

    public MultiViewLabeledDataInstance(String name, List<T> features, S label, Object source) {
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
    public LabeledDataInstance<T, S> getSingleViewDataInstance(int view) {
        return new LabeledDataInstance<>(name, features.get(view), label, source);
    }

    @Override
    protected LabeledDataInstanceBase<T, S> toDataInstanceBase() {
        return new LabeledDataInstanceBase<>(name, label, source);
    }
}
