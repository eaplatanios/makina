package org.platanios.learn.optimization.function;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * A quadratic function of the form \(f(x)=\frac{1}{2}x^TAx-b^Tx\).
 *
 * @author Emmanouil Antonios Platanios
 */
public class QuadraticFunction extends AbstractFunction {
    private final RealMatrix A;
    private final RealVector b;

    public QuadraticFunction(RealMatrix A, RealVector b) {
        this.A = A;
        this.b = b;
    }

    @Override
    public double computeValue(RealVector point) {
        return 0.5 * A.preMultiply(point).dotProduct(point) - b.dotProduct(point);
    }

    @Override
    public RealVector computeGradient(RealVector point) {
        return A.operate(point).subtract(b);
    }

    @Override
    public RealMatrix computeHessian(RealVector point) {
        return A;
    }

    public RealMatrix getA() {
        return A;
    }

    public RealVector getB() {
        return b;
    }
}
