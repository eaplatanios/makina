package org.platanios.learn.optimization.constraint;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.NonPositiveDefiniteMatrixException;
import org.platanios.learn.math.matrix.NonSymmetricMatrixException;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractConstraint {
    private int numberOfConstraintEvaluations = 0;
    private int numberOfJacobianEvaluations = 0;

    public final Vector getValue(Vector point) {
        numberOfConstraintEvaluations++;
        return computeValue(point);
    }

    abstract protected Vector computeValue(Vector point);

    public final Matrix getJacobian(Vector point) {
        numberOfJacobianEvaluations++;
        return computeJacobian(point);
    }

    abstract protected Matrix computeJacobian(Vector point);

    public abstract Vector project(Vector point)
            throws NonSymmetricMatrixException, NonPositiveDefiniteMatrixException;

    public final int getNumberOfConstraintEvaluations() {
        return numberOfConstraintEvaluations;
    }

    public final int getNumberOfJacobianEvaluations() {
        return numberOfJacobianEvaluations;
    }
}