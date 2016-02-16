package org.platanios.learn.neural.network;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class MatrixVariable extends Variable {
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

    public int rowDimension() {
        return rowDimension;
    }

    public int columnDimension() {
        return columnDimension;
    }

    // TODO: Make this function return either matrices or vectors, depending on the type of variable.
    @Override
    public Vector value(State state) {
        return state.get(this);
    }

    public Matrix valueInMatrixForm(State state) {
        return new Matrix(state.get(this).getDenseArray(), rowDimension);
    }
}
