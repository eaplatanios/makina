package org.platanios.learn.optimization.function;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;

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
    public boolean equals(Object other) {

        if (other == this) {
            return true;
        }

        if (!super.equals(other)) {
            return false;
        }

        if (!(other instanceof QuadraticFunction)) {
            return false;
        }

        QuadraticFunction rhs = (QuadraticFunction)other;
        return new EqualsBuilder()
                .append(this.A, rhs.A)
                .append(this.b, rhs.b)
                .isEquals();

    }

    @Override
    public int hashCode() {

        return new HashCodeBuilder(31, 71)
                .append(super.hashCode())
                .append(this.A)
                .append(this.b)
                .toHashCode();

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
}
