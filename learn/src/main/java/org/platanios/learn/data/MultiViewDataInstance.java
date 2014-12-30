package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;

import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class MultiViewDataInstance<T extends Vector> {
    protected final String name;
    protected final Map<Integer, T> features;

    public MultiViewDataInstance(String name) {
        this.name = name;
        this.features = null;
    }

    public MultiViewDataInstance(String name, Map<Integer, T> features) {
        this.name = name;
        this.features = features;
    }

    public String name() {
        return name;
    }

    public Map<Integer, T> features() {
        return features;
    }

    public DataInstance<T> getSingleViewDataInstance(int view) {
        return new DataInstance<>(name, features.get(view));
    }

    protected DataInstanceBase<T> toDataInstanceBase() {
        return new DataInstanceBase<>(name);
    }
}
