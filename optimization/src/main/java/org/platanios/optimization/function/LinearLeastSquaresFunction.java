package org.platanios.optimization.function;

import org.platanios.math.matrix.Matrix;
import org.platanios.math.matrix.Vector;

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

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        LinearLeastSquaresFunction that = (LinearLeastSquaresFunction) other;

        if (!super.equals(that))
            return false;
        if (!J.equals(that.J))
            return false;
        if (!y.equals(that.y))
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + J.hashCode();
        result = 31 * result + y.hashCode();
        return result;
    }
}
