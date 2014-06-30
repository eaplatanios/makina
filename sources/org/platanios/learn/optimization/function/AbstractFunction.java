package org.platanios.learn.optimization.function;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractFunction {
    private DerivativeApproximation derivativeApproximation =
            new DerivativeApproximation(this, DerivativeApproximationMethod.CENTRAL_DIFFERENCE);

    /**
     * Computes the objective function value and the constraints values at a particular point.
     *
     * @param   point   The point in which to evaluate the objective function and the constraints.
     * @return          The value of the objective function, evaluated at the given point.
     */
    abstract public double computeValue(RealVector point);

    /**
     * Computes the first derivatives of the objective function and the constraints at a particular point.
     *
     * @param   point   The point in which to evaluate the derivatives.
     * @return          The values of the first derivatives of the objective function, evaluated at the given point.
     */
    public RealVector computeGradient(RealVector point) {
        return derivativeApproximation.approximateGradient(point);
    }

    /**
     * Computes the Hessian of the objective function at a particular point.
     *
     * @param   point   The point in which to evaluate the Hessian.
     * @return          The value of the Hessian matrix of the objective function, evaluated at the given point.
     */
    public RealMatrix computeHessian(RealVector point) {
        boolean computeGradientMethodOverridden = false;
        try {
            computeGradientMethodOverridden =
                    this.getClass().getMethod("computeGradient", RealVector.class).getDeclaringClass() != AbstractFunction.class;
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        }

        if (computeGradientMethodOverridden) {
            return derivativeApproximation.approximateHessianGivenGradient(point);
        } else {
            return derivativeApproximation.approximateHessian(point);
        }
    }
}
