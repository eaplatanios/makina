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

    private QRDecomposition linearSystemMatrixQRDecomposition;

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
        if (linearSystemMatrixQRDecomposition == null) {
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
            linearSystemMatrixQRDecomposition = new QRDecomposition(linearSystemMatrix);
        }
        Vector linearSystemVector = Vectors.build(point.size() + A.getRowDimension(), point.type());
        linearSystemVector.set(0, point.size() - 1, point);
        linearSystemVector.set(point.size(), linearSystemVector.size() - 1, b);
        return linearSystemMatrixQRDecomposition.solve(linearSystemVector).get(0, point.size() - 1);
    }

    public LinearEqualityConstraint append(LinearEqualityConstraint constraint) {
        if (A.getColumnDimension() != constraint.A.getColumnDimension())
            throw new IllegalArgumentException(
                    "Trying to append linear equality constraints for differently sized vector input."
            );

        Matrix newA = new Matrix(A.getRowDimension() + constraint.A.getRowDimension(), A.getColumnDimension());
        newA.setSubMatrix(0, A.getRowDimension() - 1, 0, A.getColumnDimension() - 1, A);
        newA.setSubMatrix(A.getRowDimension(), newA.getRowDimension() - 1, 0, A.getColumnDimension() - 1, constraint.A);
        Vector newB = Vectors.build(b.size() + constraint.b.size(), b.type());
        newB.set(0, b.size() - 1, b);
        newB.set(b.size(), newB.size() - 1, constraint.b);
        return new LinearEqualityConstraint(newA, newB);
    }

    public Matrix getA() {
        return A;
    }

    public Vector getB() {
        return b;
    }
}
