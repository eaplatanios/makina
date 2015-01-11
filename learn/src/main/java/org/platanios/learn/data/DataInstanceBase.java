package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
class DataInstanceBase<T extends Vector> {
    protected final String name;

    public DataInstanceBase(String name) {
        this.name = name;
    }

    public DataInstanceBase(DataInstanceBase dataInstance) {
        this.name = dataInstance.name;
    }

    public String name() {
        return name;
    }

    public DataInstance<T> toDataInstance(T features) {
        return new DataInstance<>(name, features);
    }

    public MultiViewDataInstance<T> toMultiViewDataInstance(List<T> features) {
        return new MultiViewDataInstance<>(name, features);
    }
}
