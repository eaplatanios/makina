package org.platanios.learn.optimization.function;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * A quadratic function of the form f(x) = x'Qx + b'x.
 *
 * Created by Anthony on 6/24/14.
 */
public class QuadraticFunction implements Function {
    private final RealMatrix Q;
    private final RealVector b;

    public QuadraticFunction(RealMatrix Q, RealVector b) {
        this.Q = Q;
        this.b = b;
    }

    public double computeValue(RealVector x) {
        return Q.preMultiply(x).dotProduct(x) + b.dotProduct(x);
    }

    public RealVector computeGradient(RealVector x) {
        return Q.operate(x).mapMultiply(2).add(b);
    }

    public RealMatrix computeHessian(RealVector x) {
        return Q.scalarMultiply(2);
    }

    public RealMatrix getQ() {
        return Q;
    }

    public RealVector getB() {
        return b;
    }
}
