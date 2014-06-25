package org.platanios.learn.optimization.linesearch;

import com.google.common.base.Preconditions;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;

/**
 * @author Emmanouil Antonios Platanios
 */
public class StrongWolfeLineSearch extends IterativeLineSearch {
    private static final double MINIMUM_STEP_SIZE_CHANGE_THRESHOLD = 1e-3;

    private final double c1;
    private final double c2;
    private final double aMax;

    public StrongWolfeLineSearch(Function objective,
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

    public StrongWolfeLineSearch(Function objective,
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

    public double performLineSearch(RealVector currentPoint,
                                    RealVector currentDirection) {
        double phi0 = objective.computeValue(currentPoint);
        double phiPrime0 = objective.computeGradient(currentPoint).dotProduct(currentDirection);
        double a0 = 0;
        double a1 = currentStepSize;

        if (a1 <= 0 || a1 >= aMax) {
            a1 = aMax / 2;
        }

        boolean firstIteration = true;

        while (true) {
            RealVector a1Point = currentPoint.add(currentDirection.mapMultiply(a1));
            double phiA1 = objective.computeValue(a1Point);
            double phiA0 = objective.computeValue(currentPoint.add(currentDirection.mapMultiply(a0)));
            if (phiA1 > phi0 + c1 * a1 * phiPrime0 || (phiA1 >= phiA0 && !firstIteration)) {
                return zoom(currentPoint, currentDirection, a0, a1);
            }
            double phiPrimeA1 = objective.computeGradient(a1Point).dotProduct(currentDirection);
            if (Math.abs(phiPrimeA1) <= - c2 * phiPrime0) {
                return a1;
            }
            if (phiPrimeA1 >= 0) {
                return zoom(currentPoint, currentDirection, a1, a0);
            }
            a0 = a1;
            a1 = 2 * a1;

            if (a1 > aMax) {
                return aMax;
            }
            firstIteration = false;
        }
    }

    private double zoom(RealVector currentPoint,
                        RealVector currentDirection,
                        double aLow,
                        double aHigh) {
        double phi0 = objective.computeValue(currentPoint);
        double phiPrime0 = objective.computeGradient(currentPoint).dotProduct(currentDirection);

        while (true) {
            // Check the case where the interval has effectively been reduced to a single value (up to some precision)
            if (Math.abs(aHigh - aLow) <= MINIMUM_STEP_SIZE_CHANGE_THRESHOLD) {
                return aLow;
            }

            double aNew = performCubicInterpolation(currentPoint, currentDirection, aLow, aHigh);
            RealVector aNewPoint = currentPoint.add(currentDirection.mapMultiply(aNew));
            double phiANew = objective.computeValue(aNewPoint);
            double phiALow = objective.computeValue(currentPoint.add(currentDirection.mapMultiply(aLow)));

            if (phiANew > phi0 + c1 * aNew * phiPrime0 || phiANew >= phiALow) {
                aHigh = aNew;
            } else {
                double phiPrimeANew = objective.computeGradient(aNewPoint).dotProduct(currentDirection);
                if (Math.abs(phiPrimeANew) <= - c2 * phiPrime0) {
                    return aNew;
                }
                if (phiPrimeANew * (aHigh - aLow) >= 0) {
                    aHigh = aLow;
                }
                aLow = aNew;
            }
        }
    }

    private double performCubicInterpolation(RealVector currentPoint,
                                             RealVector currentDirection,
                                             double aLow,
                                             double aHigh) {
        RealVector newPointLow = currentPoint.add(currentDirection.mapMultiply(aLow));
        RealVector newPointHigh = currentPoint.add(currentDirection.mapMultiply(aHigh));
        double phiALow = objective.computeValue(newPointLow);
        double phiAHigh = objective.computeValue(newPointHigh);
        double phiPrimeALow = objective.computeGradient(newPointLow).dotProduct(currentDirection);
        double phiPrimeAHigh = objective.computeGradient(newPointHigh).dotProduct(currentDirection);
        double d1 = phiPrimeALow + phiPrimeAHigh - 3 * (phiALow - phiAHigh) / (aLow - aHigh);
        double d2 = Math.signum(aHigh - aLow) * Math.sqrt(Math.pow(d1, 2) - phiPrimeALow * phiPrimeAHigh);
        double aNew = aHigh - (aHigh - aLow) * (phiPrimeAHigh + d2 - d1) / (phiPrimeAHigh - phiPrimeALow + 2 * d2);

        if (aLow <= aNew && aNew <= aHigh) {
            double phiANew = objective.computeValue(currentPoint.add(currentDirection.mapMultiply(aNew)));
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
        if (Math.abs(aNew - aLow) <= MINIMUM_STEP_SIZE_CHANGE_THRESHOLD
                || Math.abs(aNew - aHigh) <= MINIMUM_STEP_SIZE_CHANGE_THRESHOLD) {
            aNew = (aLow + aHigh) / 2;
        }

        return aNew;
    }
}
