package org.platanios.learn.optimization.linesearch;

import com.google.common.base.Preconditions;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;

/**
 * @author Emmanouil Antonios Platanios
 */
public class StrongWolfeInterpolationLineSearch extends IterativeLineSearch {
    private static final int MAXIMUM_ITERATIONS_WITH_NO_OBJECTIVE_CHANGE = 10;
    private static final double MINIMUM_DISTANCE_FROM_INTERVAL_ENDPOINTS = 1e-3;

    private final double c1;
    private final double c2;
    private final double aMax;

    public StrongWolfeInterpolationLineSearch(Function objective,
                                              StepSizeInitializationMethod stepSizeInitializationMethod,
                                              double c1,
                                              double c2,
                                              double aMax) {
        super(objective, stepSizeInitializationMethod);

        Preconditions.checkArgument(c1 > 0 && c1 < 1);
        Preconditions.checkArgument(c2 > c1 && c2 < 1);
        Preconditions.checkArgument(aMax > 0);

        this.c1 = c1;
        this.c2 = c2;
        this.aMax = aMax;
    }

    public StrongWolfeInterpolationLineSearch(Function objective,
                                              StepSizeInitializationMethod stepSizeInitializationMethod,
                                              double c1,
                                              double c2,
                                              double aMax,
                                              double initialStepSize) {
        super(objective, stepSizeInitializationMethod, initialStepSize);

        Preconditions.checkArgument(c1 > 0 && c1 < 1);
        Preconditions.checkArgument(c2 > c1 && c2 < 1);
        Preconditions.checkArgument(aMax > 0);

        this.c1 = c1;
        this.c2 = c2;
        this.aMax = aMax;
    }

    public double performLineSearch(RealVector point,
                                    RealVector direction) {
        double phi0 = objective.computeValue(point);
        double phiPrime0 = objective.computeGradient(point).dotProduct(direction);
        double a0 = 0;
        double a1 = initialStepSize;

        if (a1 <= 0 || a1 >= aMax) {
            a1 = aMax / 2;
        }

        boolean firstIteration = true;

        while (true) {
            RealVector a1Point = point.add(direction.mapMultiply(a1));
            double phiA1 = objective.computeValue(a1Point);
            double phiA0 = objective.computeValue(point.add(direction.mapMultiply(a0)));
            if (phiA1 > phi0 + c1 * a1 * phiPrime0 || (phiA1 >= phiA0 && !firstIteration)) {
                return zoom(point, direction, a0, a1);
            }
            double phiPrimeA1 = objective.computeGradient(a1Point).dotProduct(direction);
            if (Math.abs(phiPrimeA1) <= - c2 * phiPrime0) {
                return a1;
            }
            if (phiPrimeA1 >= 0) {
                return zoom(point, direction, a1, a0);
            }
            a0 = a1;
            a1 = 2 * a1; // TODO: Use a more sophisticated update rule for a1

            if (a1 > aMax) {
                return aMax;
            }
            firstIteration = false;
        }
    }

    /**
     * This function "zooms in" in the interval {@code [aLow, aHigh]} and search for a step size within that interval
     * that satisfies the strong Wolfe conditions. In each iteration the interval of possible values for the step size
     * is "shrinked" and a new value to test is chosen each time using cubic interpolation.
     *
     * @param   point
     * @param   direction
     * @param   aLow                The low endpoint of the interval of possible values for the step size.
     * @param   aHigh               The high endpoint of the interval of possible values for the step size.
     * @return                      A step size value that satisfies the Wolfe conditions.
     */
    private double zoom(RealVector point,
                        RealVector direction,
                        double aLow,
                        double aHigh) {
        double phi0 = objective.computeValue(point);
        double phiPrime0 = objective.computeGradient(point).dotProduct(direction);

        // Declaring variables used in the loop that follows
        double aNew;
        double phiANew;
        double phiALow;
        double phiPrimeANew;
        RealVector aNewPoint;

        // Variables used to test for convergence of the objective function value
        double minimumObjectiveValue = Double.MAX_VALUE;
        double minimumObjectiveValueIterationNumber = -1;
        int iterationNumber = 0;

        while (true) {
            aNew = performCubicInterpolation(point, direction, aLow, aHigh);
            aNewPoint = point.add(direction.mapMultiply(aNew));
            phiANew = objective.computeValue(aNewPoint);
            phiALow = objective.computeValue(point.add(direction.mapMultiply(aLow)));

            if (phiANew > phi0 + c1 * aNew * phiPrime0 || phiANew >= phiALow) {
                aHigh = aNew;
            } else {
                phiPrimeANew = objective.computeGradient(aNewPoint).dotProduct(direction);
                if (Math.abs(phiPrimeANew) <= - c2 * phiPrime0) {
                    return aNew;
                }
                if (phiPrimeANew * (aHigh - aLow) >= 0) {
                    aHigh = aLow;
                }
                aLow = aNew;
            }

            // Check for convergence of the objective function value
            iterationNumber++;
            if (phiANew < minimumObjectiveValue) {
                minimumObjectiveValue = phiANew;
                minimumObjectiveValueIterationNumber = iterationNumber;
            } else if (iterationNumber - minimumObjectiveValueIterationNumber
                    > MAXIMUM_ITERATIONS_WITH_NO_OBJECTIVE_CHANGE) {
                return aNew;
            }
        }
    }

    private double performCubicInterpolation(RealVector point,
                                             RealVector direction,
                                             double aLow,
                                             double aHigh) {
        RealVector newPointLow = point.add(direction.mapMultiply(aLow));
        RealVector newPointHigh = point.add(direction.mapMultiply(aHigh));
        double phiALow = objective.computeValue(newPointLow);
        double phiAHigh = objective.computeValue(newPointHigh);
        double phiPrimeALow = objective.computeGradient(newPointLow).dotProduct(direction);
        double phiPrimeAHigh = objective.computeGradient(newPointHigh).dotProduct(direction);
        double d1 = phiPrimeALow + phiPrimeAHigh - 3 * (phiALow - phiAHigh) / (aLow - aHigh);
        double d2 = Math.signum(aHigh - aLow) * Math.sqrt(Math.pow(d1, 2) - phiPrimeALow * phiPrimeAHigh);
        double aNew = aHigh - (aHigh - aLow) * (phiPrimeAHigh + d2 - d1) / (phiPrimeAHigh - phiPrimeALow + 2 * d2);

        if (aLow <= aNew && aNew <= aHigh) {
            double phiANew = objective.computeValue(point.add(direction.mapMultiply(aNew)));
            if (phiALow <= phiANew) {
                if (phiALow <= phiAHigh) {
                    aNew = aLow;
                } else {
                    aNew = aHigh;
                }
            } else if (phiAHigh <= phiANew) {
                aNew = aHigh;
            }
        } else {
            if (phiALow <= phiAHigh) {
                aNew = aLow;
            } else {
                aNew = aHigh;
            }
        }

        // Ensure that the new step length is not too close to the endpoints of the interval
        if (Math.abs(aNew - aLow) <= MINIMUM_DISTANCE_FROM_INTERVAL_ENDPOINTS
                || Math.abs(aNew - aHigh) <= MINIMUM_DISTANCE_FROM_INTERVAL_ENDPOINTS) {
            aNew = (aLow + aHigh) / 2;
        }

        return aNew;
    }
}
