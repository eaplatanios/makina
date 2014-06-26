package org.platanios.learn.optimization.function;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * A quadratic function of the form f(x) = 1/2 x'Ax - b'x.
 *
 * Created by Anthony on 6/24/14.
 */
public class QuadraticFunction implements Function {
    private final RealMatrix A;
    private final RealVector b;

    public QuadraticFunction(RealMatrix A, RealVector b) {
        this.A = A;
        this.b = b;
    }

    public double computeValue(RealVector x) {
        return 0.5 * A.preMultiply(x).dotProduct(x) - b.dotProduct(x);
    }

    public RealVector computeGradient(RealVector x) {
        return A.operate(x).subtract(b);
    }

    public RealMatrix computeHessian(RealVector x) {
        return A;
    }

    public RealMatrix getA() {
        return A;
    }

    public RealVector getB() {
        return b;
    }
}
