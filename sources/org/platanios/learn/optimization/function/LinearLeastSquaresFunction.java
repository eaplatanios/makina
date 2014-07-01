package org.platanios.learn.optimization.function;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * A linear least squares function of the form \(f(x)=\frac{1}{2}\|Jx-y\|^2\).
 *
 * @author Emmanouil Antonios Platanios
 */
public class LinearLeastSquaresFunction {
    private final RealMatrix J;
    private final RealVector y;

    public LinearLeastSquaresFunction(RealMatrix J, RealVector y) {
        this.J = J;
        this.y = y;
    }

    public double computeValue(RealVector x) {
        RealVector r = J.operate(x).subtract(y);
        return 0.5 * r.dotProduct(r);
    }

    public RealVector computeGradient(RealVector x) {
        RealVector r = J.operate(x).subtract(y);
        return J.transpose().operate(r);
    }

    public RealMatrix computeHessian(RealVector x) {
        return J.transpose().multiply(J);
    }

    public RealMatrix getJ() {
        return J;
    }

    public RealVector getY() {
        return y;
    }
}
