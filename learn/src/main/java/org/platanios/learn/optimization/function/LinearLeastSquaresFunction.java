package org.platanios.learn.optimization.function;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;

/**
 * A linear least squares function of the form \(f(x)=\frac{1}{2}\|Jx-y\|^2\).
 *
 * @author Emmanouil Antonios Platanios
 */
public final class LinearLeastSquaresFunction extends AbstractLeastSquaresFunction {
    private final Matrix J;
    private final Vector y;

    public LinearLeastSquaresFunction(Matrix J, Vector y) {
        this.J = J;
        this.y = y;
    }

    @Override
    public boolean equals(Object other) {
        if (other == this) {
            return true;
        }

        if (!super.equals(other)) {
            return false;
        }

        if (!(other instanceof LinearLeastSquaresFunction)) {
            return false;
        }

        LinearLeastSquaresFunction rhs = (LinearLeastSquaresFunction)other;

        return new EqualsBuilder()
                .append(this.J, rhs.J)
                .append(this.y, rhs.y)
                .isEquals();

    }

    @Override
    public int hashCode() {

        return new HashCodeBuilder(7, 61)
                .append(super.hashCode())
                .append(this.J)
                .append(this.y)
                .toHashCode();

    }

    public Vector computeResiduals(Vector point) {
        return J.multiply(point).sub(y);
    }

    public Matrix computeJacobian(Vector point) {
        return J;
    }

    public Matrix getJ() {
        return J;
    }

    public Vector getY() {
        return y;
    }
}
