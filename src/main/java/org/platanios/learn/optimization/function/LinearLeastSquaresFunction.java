package org.platanios.learn.optimization.function;

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
