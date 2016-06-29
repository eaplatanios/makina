package makina.optimization.constraint;

import makina.math.matrix.Matrix;
import makina.math.matrix.Vector;

/**
 * A linear inequality constraint of the form \(a'x<=b\).
 *
 * @author Emmanouil Antonios Platanios
 */
public final class LinearInequalityConstraint extends AbstractInequalityConstraint {
    private final Vector a;
    private final double b;

    public LinearInequalityConstraint(Vector a, double b) {
        this.a = a;
        this.b = b;
    }

    @Override
    public double computeValue(Vector point) {
        return a.dot(point) - b;
    }

    @Override
    public Vector computeGradient(Vector point) {
        return a.copy();
    }

    @Override
    public Matrix computeHessian(Vector point) {
        return Matrix.zeros(point.size(), point.size());
    }

    public Vector getA() {
        return a;
    }

    public double getB() {
        return b;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        LinearInequalityConstraint that = (LinearInequalityConstraint) other;

        if (!super.equals(that))
            return false;
        if (!a.equals(that.a))
            return false;
        if (b != that.b)
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + a.hashCode();
        result = 31 * result + Double.hashCode(b);
        return result;
    }
}
