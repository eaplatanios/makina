package org.platanios.learn.data;

import com.google.common.base.Objects;
import org.platanios.math.matrix.Vector;

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

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        DataInstanceBase<?> that = (DataInstanceBase<?>) other;

        return Objects.equal(name, that.name);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(name);
    }
}
