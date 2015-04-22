package org.platanios.learn.optimization.linesearch;

import com.google.common.base.Preconditions;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.function.AbstractFunction;

/**
 * A collection of static methods that check whether a selected step size value satisfies certain conditions for the
 * given optimization problem.
 *
 * @author Emmanouil Antonios Platanios
 */
public final class LineSearchConditions {
    /**
     * Checks whether the Armijo condition (also known as the sufficient decrease condition) is satisfied for a given
     * objective function, point, direction and step size. The Armijo condition makes sure that the reduction in the
     * objective function value is proportional to both the step size and the directional derivative. A typical value
     * for the proportionality constant, {@code c}, is 1e-4.
     *
     * @param   objective                       The objective function instance.
     * @param   point                           The point at which we check whether the Armijo condition is satisfied.
     * @param   direction                       The direction for which we check whether the Armijo condition is
     *                                          satisfied.
     * @param   stepSize                        The value of the step size for which we check whether the Armijo
     *                                          condition is satisfied.
     * @param   c                               The proportionality constant used for the Armijo condition. The value
     *                                          provided must lie between 0 and 1.
     * @param   objectiveValueAtCurrentPoint    The value of the objective function at the current point.
     * @param   objectiveGradientAtCurrentPoint The gradient of the objective function evaluated at the current point.
     * @return                                  A boolean value indicating whether the Armijo condition is satisfied for
     *                                          the given objective, point, direction and step size.
     */
    public static boolean checkArmijoCondition(AbstractFunction objective,
                                               Vector point,
                                               Vector direction,
                                               double stepSize,
                                               double c,
                                               double objectiveValueAtCurrentPoint,
                                               Vector objectiveGradientAtCurrentPoint) {
        Preconditions.checkArgument(c > 0 && c < 1);

        double newObjectiveValue = objective.getValue(direction.mult(stepSize).add(point));

        return newObjectiveValue <=
                objectiveValueAtCurrentPoint + c * stepSize * objectiveGradientAtCurrentPoint.inner(direction);
    }

    /**
     * Checks whether the Wolfe conditions are satisfied for a given objective function, point, direction and step size.
     * The Wolfe conditions consist of the Armijo condition (also known as the sufficient decrease condition) and the
     * curvature condition. The Armijo condition makes sure that the reduction in the objective function value is
     * proportional to both the step size and the directional derivative. This condition is satisfied for all
     * sufficiently small values of the step size and so, in order to ensure that the optimization algorithm makes
     * sufficient progress, we also check for the curvature condition. Typical values for the proportionality constants
     * are: for {@code c1}, 1e-4, and for {@code c2}, 0.9 when the search direction is chosen by a Newton or
     * quasi-Newton method and 0.1 when the search direction is obtained from a nonlinear conjugate gradient method.
     *
     * @param   objective                       The objective function instance.
     * @param   point                           The point at which we check whether the Wolfe conditions are satisfied.
     * @param   direction                       The direction for which we check whether the Wolfe conditions are
     *                                          satisfied.
     * @param   stepSize                        The value of the step size for which we check whether the Wolfe
     *                                          conditions are satisfied.
     * @param   c1                              The proportionality constant used for the first of the two Wolfe
     *                                          conditions (that is, the Armijo condition). The value provided must lie
     *                                          between 0 and 1.
     * @param   c2                              The proportionality constant used for the second of the two Wolfe
     *                                          conditions (that is, the curvature condition). The value provided must
     *                                          lie between {@code c1} and 1.
     * @param   strong                          A boolean value indicating whether or not to check for the strong Wolfe
     *                                          conditions. The only difference is actually on the curvature condition
     *                                          and, when we use the strong Wolfe conditions instead of the simple Wolfe
     *                                          conditions, we effectively exclude points from the search that are far
     *                                          from the exact line search solution.
     * @param   objectiveValueAtCurrentPoint    The value of the objective function at the current point.
     * @param   objectiveGradientAtCurrentPoint The gradient of the objective function evaluated at the current point.
     * @return                                  A boolean value ind objective, point, direction and step size.
     */
    public static boolean checkWolfeConditions(AbstractFunction objective,
                                               Vector point,
                                               Vector direction,
                                               double stepSize,
                                               double c1,
                                               double c2,
                                               boolean strong,
                                               double objectiveValueAtCurrentPoint,
                                               Vector objectiveGradientAtCurrentPoint) {
        Preconditions.checkArgument(c1 > 0 && c1 < 1);
        Preconditions.checkArgument(c2 > c1 && c2 < 1);

        // Check the Armijo condition
        boolean armijoConditionSatisfied = checkArmijoCondition(
                objective,
                point,
                direction,
                stepSize,
                c1,
                objectiveValueAtCurrentPoint,
                objectiveGradientAtCurrentPoint
        );

        // Check the curvature condition
        double leftTerm = objective.getGradient(point.add(direction.mult(stepSize))).inner(direction);
        double rightTerm = objectiveGradientAtCurrentPoint.inner(direction);
        boolean curvatureConditionSatisfied;
        if (strong) {
            curvatureConditionSatisfied = Math.abs(leftTerm) <= c2 * Math.abs(rightTerm);
        } else {
            curvatureConditionSatisfied = leftTerm >= c2 * rightTerm;
        }

        return armijoConditionSatisfied && curvatureConditionSatisfied;
    }

    /**
     * Checks whether the Goldstein conditions are satisfied for a given objective function, point, direction and step
     * size. The Goldstein conditions are similar to the Wolfe conditions and they can also be stated as a pair of
     * inequalities: one inequality corresponding to the Armijo condition (also known as the sufficient decrease
     * condition) and another inequality used to bound the step size from below. However, that second inequality in the
     * case of the Goldstein conditions might exclude all points from the search that are solutions to the exact line
     * search problem. The Goldstein conditions are often used in Newton-type methods but are not well suited for
     * quasi-Newton methods that maintain a positive definite Hessian approximation.
     *
     * @param   objective                       The objective function instance.
     * @param   point                           The point at which we check whether the Goldstein conditions are
     *                                          satisfied.
     * @param   direction                       The direction for which we check whether the Goldstein conditions are
     *                                          satisfied.
     * @param   stepSize                        The value of the step size for which we check whether the Goldstein
     *                                          conditions are satisfied.
     * @param   c                               The proportionality constant used for the Goldstein conditions. The
     *                                          value provided must lie between 0 and 0.5.
     * @param   objectiveValueAtCurrentPoint    The value of the objective function at the current point.
     * @param   objectiveGradientAtCurrentPoint The gradient of the objective function evaluated at the current point.
     * @return                                  A boolean value indicating whether the Goldstein conditions are
     *                                          satisfied for the given objective, point, direction and step size.
     */
    public static boolean checkGoldsteinConditions(AbstractFunction objective,
                                                   Vector point,
                                                   Vector direction,
                                                   double stepSize,
                                                   double c,
                                                   double objectiveValueAtCurrentPoint,
                                                   Vector objectiveGradientAtCurrentPoint) {
        Preconditions.checkArgument(c > 0 && c < 0.5);

        double newObjectiveValue = objective.getValue(point.add(direction.mult(stepSize)));
        double scaledSearchDirection = stepSize * objectiveGradientAtCurrentPoint.inner(direction);
        double lowerBound = objectiveValueAtCurrentPoint + (1 - c) * scaledSearchDirection;
        double upperBound = objectiveValueAtCurrentPoint + c * scaledSearchDirection;

        return lowerBound <= newObjectiveValue && newObjectiveValue <= upperBound;
    }
}
