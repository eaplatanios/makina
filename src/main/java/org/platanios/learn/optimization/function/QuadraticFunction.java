package org.platanios.learn.optimization.function;

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
    public double computeValue(Vector point) {
        return 0.5 * point.mult(A).inner(point) - b.inner(point);
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
