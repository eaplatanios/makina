package org.platanios.learn.neural.network;

import com.google.common.base.Objects;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class VectorVariable extends Variable {
    VectorVariable(int id, int size) {
        super(id, size);
    }

    VectorVariable(int id, String name, int size) {
        super(id, name, size);
    }

    @Override
    public Vector value(State state) {
        return state.get(this);
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        VectorVariable that = (VectorVariable) other;

        return Objects.equal(id, that.id)
                && Objects.equal(size, that.size);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(id, size);
    }
}
