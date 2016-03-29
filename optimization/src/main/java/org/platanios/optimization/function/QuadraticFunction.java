package org.platanios.optimization.function;

import org.platanios.math.matrix.Matrix;
import org.platanios.math.matrix.Vector;

/**
 * A quadratic function of the form \(f(x)=\frac{1}{2}x^TAx-b^Tx\).
 *
 * @author Emmanouil Antonios Platanios
 */
public final class QuadraticFunction extends AbstractFunction {
    private final Matrix A;
    private final Vector b;

    public QuadraticFunction(Matrix A, Vector b) {
        this.A = A;
        this.b = b;
    }

    @Override
    public double computeValue(Vector point) {
        return 0.5 * point.transMult(A).inner(point) - b.inner(point);
    }

    @Override
    public Vector computeGradient(Vector point) {
        return A.multiply(point).sub(b);
    }

    @Override
    public Matrix computeHessian(Vector point) {
        return A;
    }

    public Matrix getA() {
        return A;
    }

    public Vector getB() {
        return b;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        QuadraticFunction that = (QuadraticFunction) other;

        if (!super.equals(that))
            return false;
        if (!A.equals(that.A))
            return false;
        if (!b.equals(that.b))
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + A.hashCode();
        result = 31 * result + b.hashCode();
        return result;
    }
}
