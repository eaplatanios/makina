package org.platanios.learn.optimization.linesearch;

import com.google.common.base.Preconditions;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;

/**
 * Implements an interpolation based line search algorithm that returns a step size value that satisfies the Armijo
 * condition (also known as the sufficient decrease condition). This is an implementation of an algorithm described in
 * pages 57-58 of the book "Numerical Optimization", by Jorge Nocedal and Stephen Wright.
 *
 * @author Emmanouil Antonios Platanios
 */
public class ArmijoInterpolationLineSearch extends IterativeLineSearch {
    /** Threshold for the minimum allowed step size change during an interpolation step. */
    private static final double MINIMUM_STEP_SIZE_CHANGE_THRESHOLD = 1e-3;
    /** Threshold for the minimum allowed step size change ratio during an interpolation step. */
    private static final double MINIMUM_STEP_SIZE_RATIO_THRESHOLD = 1e-1;

    /** The proportionality constant used for the Armijo condition. */
    private final double c;

    /** Used to store the old value of the step size during the iterative line search procedure. */
    private double aOld;
    /** Used to store the new value of the step size during the iterative line search procedure. */
    private double aNew;

    /**
     * Constructs an Armijo interpolation based line search solver for the provided objective function instance, using
     * the provided step size initialization method to compute the initial value of the step size and the provided
     * parameters. If the selected step size initialization method is simply a constant value, then the alternative
     * constructor must be used that receives the initial step size value as its second argument.
     *
     * @param   objective                       The objective function instance.
     * @param   stepSizeInitializationMethod    The step size initialization method.
     * @param   c                               The proportionality constant used for the Armijo condition.
     */
    public ArmijoInterpolationLineSearch(Function objective,
                                         StepSizeInitializationMethod stepSizeInitializationMethod,
                                         double c) {
        super(objective, stepSizeInitializationMethod);

        Preconditions.checkArgument(c > 0 && c < 1);

        this.c = c;
    }

    /**
     * Constructs an Armijo interpolation based line search solver for the provided objective function instance, using
     * the provided initial value for the step size and the provided parameters. If another step size initialization
     * method is required, then the alternative constructor must be used that receives the step size initialization
     * method as its second argument.
     *
     * @param   objective           The objective function instance.
     * @param   initialStepSize     The initial step size value to use (it must have a value greater than zero).
     * @param   c                   The proportionality constant used for the Armijo condition.
     */
    public ArmijoInterpolationLineSearch(Function objective,
                                         double initialStepSize,
                                         double c) {
        super(objective, initialStepSize);

        Preconditions.checkArgument(c > 0 && c < 1);
        Preconditions.checkArgument(initialStepSize > 0);

        this.c = c;
    }

    /**
     * Constructs an Armijo interpolation based line search solver for the provided objective function instance, using
     * the provided step size initialization method, the provided initial step size and the provided parameters. Using
     * this constructor only makes sense if the selected step size initialization method is CONSTANT. For all other
     * cases the extra initial step size argument is not required. Furthermore, if the selected step size initialization
     * method is not CONSTANT, then that extra argument is completely ignored!
     *
     * @param   objective                       The objective function instance.
     * @param   stepSizeInitializationMethod    The step size initialization method.
     * @param   c                               The proportionality constant used for the Armijo condition.
     * @param   initialStepSize                 The initial step size value to use (it must have a value greater than
     *                                          zero).
     */
    public ArmijoInterpolationLineSearch(Function objective,
                                         StepSizeInitializationMethod stepSizeInitializationMethod,
                                         double c,
                                         double initialStepSize) {
        super(objective, stepSizeInitializationMethod, initialStepSize);

        Preconditions.checkArgument(c > 0 && c < 1);
        Preconditions.checkArgument(initialStepSize > 0);

        this.c = c;
    }

    /**
     * {@inheritDoc}
     *
     * @return  A step size value that satisfies the Armijo condition (also known as the sufficient decrease condition).
     */
    public double performLineSearch(RealVector point,
                                    RealVector direction) {
        double phi0 = objective.computeValue(point);
        RealVector objectiveGradientAtCurrentPoint = objective.computeGradient(point);
        double phiPrime0 = objectiveGradientAtCurrentPoint.dotProduct(direction);

        aNew = initialStepSize;
        boolean firstIteration = true;

        while (!LineSearchConditions.checkArmijoCondition(objective,
                                                          point,
                                                          direction,
                                                          aNew,
                                                          c,
                                                          phi0,
                                                          objectiveGradientAtCurrentPoint)) {
            if (firstIteration) {
                performQuadraticInterpolation(point,
                                              direction,
                                              phi0,
                                              phiPrime0);
                firstIteration = false;
            } else {
                performCubicInterpolation(point,
                                          direction,
                                          phi0,
                                          phiPrime0);
            }
        }

        return aNew;
    }

    /**
     * Performs a quadratic interpolation using the available information in order to obtain an approximation of the
     * &phi; function and returns the step size value that minimizes that approximation. This function is only used for
     * the first iteration of the line search algorithm, when we do not yet have enough information available to perform
     * a cubic interpolation.
     *
     * @param   point       The point at which we perform the line search.
     * @param   direction   The direction for which we perform the line search.
     * @param   phi0        The value of &phi;(0) (that is, the value of the objective function at the point at which we
     *                      perform the line search).
     * @param   phiPrime0   The value of &phi;'(0) (that is, the value of the objective function gradient at the point
     *                      at which we perform the line search).
     */
    private void performQuadraticInterpolation(RealVector point,
                                               RealVector direction,
                                               double phi0,
                                               double phiPrime0) {
        aOld = aNew;
        double phiA0 = objective.computeValue(point.add(direction.mapMultiply(aOld)));
        aNew = -phiPrime0 * Math.pow(aOld, 2) / (2 * (phiA0 - phi0 - aOld * phiPrime0));

        // Ensure that we make reasonable progress and that the final step size is not too small.
        if (Math.abs(aNew - aOld) <= MINIMUM_STEP_SIZE_CHANGE_THRESHOLD
                || aNew / aOld <= MINIMUM_STEP_SIZE_RATIO_THRESHOLD) {
            aNew = aOld / 2;
        }
    }

    /**
     * Performs a cubic interpolation using the available information in order to obtain an approximation of the &phi;
     * function and returns the step size value that minimizes that approximation.
     *
     * @param   point       The point at which we perform the line search.
     * @param   direction   The direction for which we perform the line search.
     * @param   phi0        The value of &phi;(0) (that is, the value of the objective function at the point at which we
     *                      perform the line search).
     * @param   phiPrime0   The value of &phi;'(0) (that is, the value of the objective function gradient at the point
     *                      at which we perform the line search).
     */
    private void performCubicInterpolation(RealVector point,
                                           RealVector direction,
                                           double phi0,
                                           double phiPrime0) {
        double a0Square = Math.pow(aOld, 2);
        double a1Square = Math.pow(aNew, 2);
        double a0Cube = Math.pow(aOld, 3);
        double a1Cube = Math.pow(aNew, 3);
        double phiA0 = objective.computeValue(point.add(direction.mapMultiply(aOld)));
        double phiA1 = objective.computeValue(point.add(direction.mapMultiply(aNew)));
        double denominator = a0Square * a1Square * (aNew - aOld);
        double a = (a0Square * (phiA1 - phi0 - aNew * phiPrime0) - a1Square * (phiA0 - phi0 - aOld * phiPrime0))
                / denominator;
        double b = (-a0Cube * (phiA1 - phi0 - aNew * phiPrime0) + a1Cube * (phiA0 - phi0 - aOld * phiPrime0))
                / denominator;

        aOld = aNew;
        aNew = -(b - Math.sqrt(Math.pow(b, 2) - 3 * a * phiPrime0)) / (3 * a);

        // Ensure that we make reasonable progress and that the final step size is not too small.
        if (Math.abs(aNew - aOld) <= MINIMUM_STEP_SIZE_CHANGE_THRESHOLD
                || aNew / aOld <= MINIMUM_STEP_SIZE_RATIO_THRESHOLD) {
            aNew = aOld / 2;
        }
    }
}
