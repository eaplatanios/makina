package org.platanios.learn.optimization.constraint;

import org.platanios.learn.math.MathUtilities;
import org.platanios.learn.math.matrix.*;
import org.platanios.learn.serialization.UnsafeSerializationUtilities;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractConstraint {
    protected final double epsilon = MathUtilities.computeMachineEpsilonDouble();

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
            throws SingularMatrixException;

    public final int getNumberOfConstraintEvaluations() {
        return numberOfConstraintEvaluations;
    }

    public final int getNumberOfJacobianEvaluations() {
        return numberOfJacobianEvaluations;
    }

    public static AbstractConstraint build(InputStream inputStream) throws IOException {
        ConstraintType constraintType = ConstraintType.values()[UnsafeSerializationUtilities.readInt(inputStream)];
        return constraintType.buildConstraint(inputStream, false);
    }
}
