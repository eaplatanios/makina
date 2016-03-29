package org.platanios.learn.data;

import org.platanios.math.matrix.Vector;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class MultiViewDataInstance<T extends Vector> {
    protected final String name;
    protected final List<T> features;

    public MultiViewDataInstance(String name, List<T> features) {
        this.name = name;
        this.features = features;
    }

    public String name() {
        return name;
    }

    public List<T> features() {
        return features;
    }

    public DataInstance<T> getSingleViewDataInstance(int view) {
        return new DataInstance<>(name, features.get(view));
    }

    protected DataInstanceBase<T> toDataInstanceBase() {
        return new DataInstanceBase<>(name);
    }
}
