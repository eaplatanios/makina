package org.platanios.learn.data;

import com.google.common.base.Objects;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataInstance<T extends Vector> {
    protected final String name;
    protected final T features;

    public DataInstance(String name) {
        this.name = name;
        this.features = null;
    }

    public DataInstance(String name, T features) {
        this.name = name;
        this.features = features;
    }

    public String name() {
        return name;
    }

    public T features() {
        return features;
    }

    protected DataInstanceBase<T> toDataInstanceBase() {
        return new DataInstanceBase<>(name);
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        DataInstance<?> that = (DataInstance<?>) other;

        return Objects.equal(name, that.name)
                && Objects.equal(features, that.features);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(name, features);
    }
}
