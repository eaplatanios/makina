package org.platanios.learn.optimization.constraint;

import org.platanios.learn.math.matrix.*;

/**
 * A linear equality constraint of the form \(Ax=b\).
 *
 * @author Emmanouil Antonios Platanios
 */
public final class LinearEqualityConstraint extends AbstractEqualityConstraint {
    private final Matrix A;
    private final Vector b;

    private QRDecomposition linearSystemMatrixSingularValueDecomposition;

    public LinearEqualityConstraint(Matrix A, Vector b) {
        this.A = A;
        this.b = b;
    }

    public LinearEqualityConstraint(Vector a, double b) {
        this.A = new Matrix(a.getDenseArray(), 1);
        this.b = Vectors.dense(1, b);
    }

    @Override
    public Vector computeValue(Vector point) {
        return A.multiply(point).sub(b);
    }

    @Override
    public Matrix computeJacobian(Vector point) {
        return A;
    }

    @Override
    public Vector project(Vector point) throws SingularMatrixException {
        if (linearSystemMatrixSingularValueDecomposition == null) {
            Matrix linearSystemMatrix = new Matrix(A.getRowDimension() + A.getColumnDimension(),
                                             A.getRowDimension() + A.getColumnDimension());
            linearSystemMatrix.setSubMatrix(0,
                                      A.getColumnDimension() - 1,
                                      0,
                                      A.getColumnDimension() - 1,
                                      Matrix.generateIdentityMatrix(A.getColumnDimension()));
            linearSystemMatrix.setSubMatrix(0,
                                      A.getColumnDimension() - 1,
                                      A.getColumnDimension(),
                                      linearSystemMatrix.getColumnDimension() - 1,
                                      A.transpose());
            linearSystemMatrix.setSubMatrix(A.getColumnDimension(),
                                      linearSystemMatrix.getRowDimension() - 1,
                                      0,
                                      A.getColumnDimension() - 1,
                                      A);
            linearSystemMatrixSingularValueDecomposition = new QRDecomposition(linearSystemMatrix);
        }
        Vector linearSystemVector = Vectors.build(point.size() + A.getRowDimension(), point.type());
        linearSystemVector.set(0, point.size() - 1, point);
        linearSystemVector.set(point.size(), linearSystemVector.size() - 1, b);
        return linearSystemMatrixSingularValueDecomposition.solve(linearSystemVector).get(0, point.size() - 1);
    }

    public Matrix getA() {
        return A;
    }

    public Vector getB() {
        return b;
    }
}
