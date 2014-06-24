package org.platanios.learn.optimization.function;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface Function {
    /**
     * Computes the objective function value and the constraints values at a particular point.
     *
     * @param   optimizationVariables   The point in which to evaluate the objective function and the constraints.
     * @return                          The value of the objective function, evaluated at the given point.
     */
    double computeValue(RealVector optimizationVariables);

    /**
     * Computes the first derivatives of the objective function and the constraints at a particular point.
     *
     * @param   optimizationVariables   The point in which to evaluate the derivatives.
     * @return                          The values of the first derivatives of the objective function, evaluated at the
     *                                  given point.
     */
    default RealVector computeGradient(RealVector optimizationVariables) {
        // TODO: Implement numerical differentiation methods.
        throw new NotImplementedException();
    }

    /**
     * Computes the Hessian of the objective function at a particular point.
     *
     * @param   optimizationVariables   The point in which to evaluate the Hessian.
     * @return                          The value of the Hessian matrix of the objective function, evaluated at the
     *                                  given point.
     */
    default RealMatrix computeHessian(RealVector optimizationVariables) {
        // TODO: Implement numerical Hessian computation methods and quasi-Newton methods.
        throw new NotImplementedException();
    }
}
