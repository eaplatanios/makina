package org.platanios.learn.optimization.linesearch;

import com.google.common.base.Preconditions;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.function.AbstractFunction;

/**
 * Implements an interpolation based line search algorithm that returns a step size value that satisfies the Armijo
 * condition (also known as the sufficient decrease condition). This is an implementation of an algorithm described in
 * pages 57-58 of the book "Numerical Optimization", by Jorge Nocedal and Stephen Wright.
 *
 * @author Emmanouil Antonios Platanios
 */
public final class ArmijoInterpolationLineSearch extends IterativeLineSearch {
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
     * the provided parameter for the Armijo condition.
     *
     * @param   objective   The objective function instance.
     * @param   c           The proportionality constant used for the Armijo condition. The value provided must lie
     *                      between 0 and 1.
     */
    public ArmijoInterpolationLineSearch(AbstractFunction objective, double c) {
        super(objective);

        Preconditions.checkArgument(c > 0 && c < 1);

        this.c = c;
    }

    /**
     * {@inheritDoc}
     *
     * @return  A step size value that satisfies the Armijo condition (also known as the sufficient decrease condition).
     */
    @Override
    public double performLineSearch(Vector point,
                                    Vector direction) {
        double phi0 = objective.getValue(point);
        Vector objectiveGradientAtCurrentPoint = objective.getGradient(point);
        double phiPrime0 = objectiveGradientAtCurrentPoint.inner(direction);

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
     */
    private void performQuadraticInterpolation(Vector point,
                                               Vector direction,
                                               double phi0,
                                               double phiPrime0) {
        aOld = aNew;
        double phiA0 = objective.getValue(point.add(direction.mult(aOld)));
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
     */
    private void performCubicInterpolation(Vector point,
                                           Vector direction,
                                           double phi0,
                                           double phiPrime0) {
        double a0Square = Math.pow(aOld, 2);
        double a1Square = Math.pow(aNew, 2);
        double a0Cube = Math.pow(aOld, 3);
        double a1Cube = Math.pow(aNew, 3);
        double phiA0 = objective.getValue(point.add(direction.mult(aOld)));
        double phiA1 = objective.getValue(point.add(direction.mult(aNew)));
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
