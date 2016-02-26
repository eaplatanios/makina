package org.platanios.learn.neural.network;

import com.google.common.base.Objects;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
class ConstantVectorVariable extends VectorVariable {
    private final Vector value;

    ConstantVectorVariable(int id, Vector value) {
        super(id, value.size());
        this.value = value;
    }

    @Override
    Vector value(NetworkState state) {
        return value;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        ConstantVectorVariable that = (ConstantVectorVariable) other;

        return Objects.equal(id, that.id)
                && Objects.equal(name, that.name)
                && Objects.equal(size, that.size)
                && Objects.equal(value, that.value);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(id, name, size, value);
    }
}
