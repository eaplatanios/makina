package org.platanios.learn.optimization.function;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * A linear least squares function of the form \(f(x)=\frac{1}{2}\|Jx-y\|^2\).
 *
 * @author Emmanouil Antonios Platanios
 */
public class LinearLeastSquaresFunction extends AbstractLeastSquaresFunction {
    private final RealMatrix J;
    private final RealVector y;

    public LinearLeastSquaresFunction(RealMatrix J, RealVector y) {
        this.J = J;
        this.y = y;
    }

    public RealVector computeResiduals(RealVector point) {
        return J.operate(point).subtract(y);
    }

    public RealMatrix computeJacobian(RealVector point) {
        return J;
    }

    public RealMatrix getJ() {
        return J;
    }

    public RealVector getY() {
        return y;
    }
}
