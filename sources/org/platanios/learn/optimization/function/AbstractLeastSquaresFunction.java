package org.platanios.learn.optimization.function;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractLeastSquaresFunction extends AbstractFunction {
    public double computeValue(RealVector point) {
        RealVector residuals = computeResiduals(point);
        return 0.5 * residuals.dotProduct(residuals);
    }

    public RealVector computeGradient(RealVector point) {
        RealVector residuals = computeResiduals(point);
        RealMatrix jacobian = computeJacobian(point);
        return jacobian.transpose().operate(residuals);
    }

    public RealMatrix computeHessian(RealVector point) {
        RealMatrix jacobian = computeJacobian(point);
        return jacobian.transpose().multiply(jacobian);
    }

    public abstract RealVector computeResiduals(RealVector point);
    public abstract RealMatrix computeJacobian(RealVector point);
}
