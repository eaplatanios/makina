package org.platanios.learn.optimization;

import com.google.common.base.Preconditions;
import org.apache.commons.math3.linear.RealVector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LineSearchConditions {
    /**
     * Checks whether the Armijo condition (also known as the sufficient decrease condition) is satisfied for a given
     * step size and descent direction. The Armijo condition makes sure that the reduction in the objective function
     * value is proportional to both the step size and the directional derivative. A typical value for the
     * proportionality constant, {@code c}, is 1e-4.
     *
     * @param   objectiveFunction
     * @param   currentPoint
     * @param   direction
     * @param   stepSize
     * @param   c
     * @param   objectiveFunctionValueAtCurrentPoint
     * @param   objectiveFunctionGradientAtCurrentPoint
     * @return
     */
    public static boolean checkArmijoCondition(ObjectiveFunction objectiveFunction,
                                               RealVector currentPoint,
                                               RealVector direction,
                                               double stepSize,
                                               double c,
                                               double objectiveFunctionValueAtCurrentPoint,
                                               RealVector objectiveFunctionGradientAtCurrentPoint) {
        Preconditions.checkArgument(c > 0 && c < 1);

        double newObjectiveFunctionValue =
                objectiveFunction.computeValue(direction.mapMultiply(stepSize).add(currentPoint));
        double upperBound = objectiveFunctionValueAtCurrentPoint
                + c * stepSize * objectiveFunctionGradientAtCurrentPoint.dotProduct(direction);

        return newObjectiveFunctionValue <= upperBound;
    }

    /**
     * Checks whether the Wolfe conditions are satisfied for a given step size, direction and objective function. The
     * Wolfe conditions consist of the Armijo condition (also known as the sufficient decrease condition) and the
     * curvature condition. The Armijo condition makes sure that the reduction in the objective function value is
     * proportional to both the step size and the directional derivative. This condition is satisfied for all
     * sufficiently small values of the step size and so, in order to ensure that the optimization algorithm makes
     * sufficient progress, we also check for the curvature condition. Typical values for the proportionality constants
     * are: for c1, 1e-4, and for c2, 0.9 when the search direction is chosen by a Newton or quasi-Newton method and
     * 0.1 when the search direction is obtained from a nonlinear conjugate gradient method.
     *
     * @param   objectiveFunction
     * @param   currentPoint
     * @param   direction
     * @param   stepSize
     * @param   c1
     * @param   c2
     * @param   strong              The different with the curvature condition of the simple Wolfe conditions is that in
     *                              this case we exclude points from the search that are far from the exact line search
     *                              solution.
     * @return
     */
    public static boolean checkWolfeConditions(ObjectiveFunctionWithGradient objectiveFunction,
                                               RealVector currentPoint,
                                               RealVector direction,
                                               double stepSize,
                                               double c1,
                                               double c2,
                                               boolean strong) {
        Preconditions.checkArgument(c1 > 0 && c1 < 1);
        Preconditions.checkArgument(c2 > c1 && c2 < 1);

        double objectiveFunctionValueAtCurrentPoint = objectiveFunction.computeValue(currentPoint);
        RealVector objectiveFunctionGradientAtCurrentPoint = objectiveFunction.computeGradient(currentPoint);

        // Check the Armijo condition
        boolean armijoConditionSatisfied = checkArmijoCondition(
                objectiveFunction,
                currentPoint,
                direction,
                stepSize,
                c1,
                objectiveFunctionValueAtCurrentPoint,
                objectiveFunctionGradientAtCurrentPoint
        );

        // Check the curvature condition
        double curvatureConditionTerm1 = objectiveFunction.computeGradient(
                currentPoint.add(direction.mapMultiply(stepSize))
        ).dotProduct(direction);
        double curvatureConditionTerm2 = objectiveFunctionGradientAtCurrentPoint.dotProduct(direction);
        boolean curvatureConditionSatisfied;
        if (strong) {
            curvatureConditionSatisfied = Math.abs(curvatureConditionTerm1) >= c2 * Math.abs(curvatureConditionTerm2);
        } else {
            curvatureConditionSatisfied = curvatureConditionTerm1 >= c2 * curvatureConditionTerm2;
        }

        return armijoConditionSatisfied && curvatureConditionSatisfied;
    }

    /**
     * Checks whether the Goldstein conditions are satisfied for a given step size, direction and objective function.
     * The Goldstein conditions are similar to the Wolfe conditions and they can also be stated as a pair of
     * inequalities: one inequality corresponding to the Armijo condition (also known as the sufficient decrease
     * condition) and another inequality used to bound step size from below. However, that second inequality in the case
     * of the Goldstein conditions might exclude all points from the search that are solutions to the exact line search
     * problem. The Goldstein conditions are often used in Newton-type methods but are not well suited for quasi-Newton
     * methods that maintain a positive definite Hessian approximation.
     *
     * @param   objectiveFunction
     * @param   currentPoint
     * @param   direction
     * @param   stepSize
     * @param   c
     * @return
     */
    public static boolean checkGoldsteinConditions(ObjectiveFunctionWithGradient objectiveFunction,
                                                   RealVector currentPoint,
                                                   RealVector direction,
                                                   double stepSize,
                                                   double c) {
        Preconditions.checkArgument(c > 0 && c < 0.5);

        double objectiveFunctionValueAtCurrentPoint = objectiveFunction.computeValue(currentPoint);
        RealVector objectiveFunctionGradientAtCurrentPoint = objectiveFunction.computeGradient(currentPoint);

        double newObjectiveFunctionValue =
                objectiveFunction.computeValue(currentPoint.add(direction.mapMultiply(stepSize)));
        double scaledSearchDirection = stepSize * objectiveFunctionGradientAtCurrentPoint.dotProduct(direction);
        double lowerBound = objectiveFunctionValueAtCurrentPoint + (1 - c) * scaledSearchDirection;
        double upperBound = objectiveFunctionValueAtCurrentPoint + c * scaledSearchDirection;

        return lowerBound <= newObjectiveFunctionValue && newObjectiveFunctionValue <= upperBound;
    }
}
