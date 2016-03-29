package org.platanios.learn.neural.network;

import com.google.common.base.Objects;
import org.platanios.math.matrix.Matrix;
import org.platanios.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
class MatrixVariable extends Variable {
    private final int rowDimension;
    private final int columnDimension;

    MatrixVariable(int id, int rowDimension, int columnDimension) {
        super(id, rowDimension * columnDimension);
        this.rowDimension = rowDimension;
        this.columnDimension = columnDimension;
    }

    MatrixVariable(int id, String name, int rowDimension, int columnDimension) {
        super(id, name, rowDimension * columnDimension);
        this.rowDimension = rowDimension;
        this.columnDimension = columnDimension;
    }

    int rowDimension() {
        return rowDimension;
    }

    int columnDimension() {
        return columnDimension;
    }

    // TODO: Make this function return either matrices or vectors, depending on the type of variable.
    @Override
    Vector value(NetworkState state) {
        return state.get(this);
    }

    Matrix valueInMatrixForm(NetworkState state) {
        return new Matrix(state.get(this).getDenseArray(), rowDimension);
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        MatrixVariable that = (MatrixVariable) other;

        return Objects.equal(id, that.id)
                && Objects.equal(name, that.name)
                && Objects.equal(rowDimension, that.rowDimension)
                && Objects.equal(columnDimension, that.columnDimension);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(id, name, rowDimension, columnDimension);
    }
}
