package org.platanios.learn.optimization.function;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;

/**
 * A quadratic function of the form \(f(x)=a^\top x+b\).
 *
 * @author Emmanouil Antonios Platanios
 */
public final class LinearFunction extends AbstractFunction {
    private final Vector a;
    private final double b;

    public LinearFunction(Vector a, double b) {
        this.a = a;
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

        if (!(other instanceof LinearFunction)) {
            return false;
        }

        LinearFunction rhs = (LinearFunction)other;

        return new EqualsBuilder()
                .append(this.a, rhs.a)
                .append(this.b, rhs.b)
                .isEquals();

    }

    @Override
    public int hashCode() {

        return new HashCodeBuilder(19, 23)
                .append(super.hashCode())
                .append(this.a)
                .append(this.b)
                .toHashCode();

    }

    @Override
    public double computeValue(Vector point) {
        return a.dot(point) + b;
    }

    @Override
    public Vector computeGradient(Vector point) {
        return a;
    }

    @Override
    public Matrix computeHessian(Vector point) {
        return new Matrix(point.size(), point.size());
    }

    public LinearFunction add(LinearFunction linearFunction) {
        if (a.size() != linearFunction.a.size())
            throw new IllegalArgumentException(
                    "Trying to add two linear functions for differently sized vector inputs."
            );

        Vector newA = a.copy();
        newA.addInPlace(linearFunction.a);
        return new LinearFunction(newA, b + linearFunction.b);
    }

    public Vector projectToHyperplane(Vector point) {
        return point.sub(a.mult((a.dot(point) + b) / a.dot(a)));
    }

    public Vector getA() {
        return a;
    }

    public double getB() {
        return b;
    }
}
