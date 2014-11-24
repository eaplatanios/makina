package org.platanios.learn.optimization.function;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractLeastSquaresFunction extends AbstractFunction {
    @Override
    public final double computeValue(Vector point) {
        Vector residuals = computeResiduals(point);
        return 0.5 * residuals.inner(residuals);
    }

    @Override
    public final Vector computeGradient(Vector point) {
        Vector residuals = computeResiduals(point);
        Matrix jacobian = computeJacobian(point);
        return jacobian.transpose().multiply(residuals);
    }

    @Override
    public final Matrix computeHessian(Vector point) {
        Matrix jacobian = computeJacobian(point);
        return jacobian.transpose().multiply(jacobian);
    }

    public abstract Vector computeResiduals(Vector point);
    public abstract Matrix computeJacobian(Vector point);
}