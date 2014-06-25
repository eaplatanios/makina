package org.platanios.learn.optimization.linesearch;

import com.google.common.base.Preconditions;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ArmijoInterpolationLineSearch extends IterativeLineSearch {
    private static final double MINIMUM_STEP_SIZE_CHANGE_THRESHOLD = 1e-3;
    private static final double MINIMUM_STEP_SIZE_RATIO_THRESHOLD = 1e-1;

    private final double c;

    private double[] mostRecentStepSizes; // [0] is the previous one and [1] is the one before that one

    public ArmijoInterpolationLineSearch(Function objective,
                                         StepSizeInitializationMethod stepSizeInitializationMethod,
                                         double c) {
        super(objective, stepSizeInitializationMethod);
        Preconditions.checkArgument(c > 0 && c < 1);
        this.c = c;
        mostRecentStepSizes = new double[2];
    }

    public ArmijoInterpolationLineSearch(Function objective,
                                         StepSizeInitializationMethod stepSizeInitializationMethod,
                                         double c,
                                         double initialStepSize) {
        super(objective, stepSizeInitializationMethod, initialStepSize);
        Preconditions.checkArgument(c > 0 && c < 1);
        Preconditions.checkArgument(initialStepSize > 0);
        this.c = c;
        mostRecentStepSizes = new double[2];
    }

    public void performLineSearch(RealVector currentPoint,
                                   RealVector currentDirection) {
        double objectiveValueAtCurrentPoint = objective.computeValue(currentPoint);
        RealVector objectiveGradientAtCurrentPoint = objective.computeGradient(currentPoint);
        double dotProductOfObjectiveGradientAndDirection = objectiveGradientAtCurrentPoint.dotProduct(currentDirection);

        mostRecentStepSizes[0] = currentStepSize;
        boolean firstIteration = true;

        while (!LineSearchConditions.checkArmijoCondition(objective,
                currentPoint,
                currentDirection,
                mostRecentStepSizes[0],
                c,
                objectiveValueAtCurrentPoint,
                objectiveGradientAtCurrentPoint)) {
            if (firstIteration) {
                performQuadraticInterpolation(
                        currentPoint,
                        currentDirection,
                        objectiveValueAtCurrentPoint,
                        dotProductOfObjectiveGradientAndDirection
                );
                firstIteration = false;
            } else {
                performCubicInterpolation(
                        currentPoint,
                        currentDirection,
                        objectiveValueAtCurrentPoint,
                        dotProductOfObjectiveGradientAndDirection
                );
            }
        }

        currentStepSize = mostRecentStepSizes[0];
    }

    private void performQuadraticInterpolation(RealVector currentPoint,
                                               RealVector currentDirection,
                                               double phi0,
                                               double phiPrime0) {
        double a0 = mostRecentStepSizes[0];
        double phiA0 = objective.computeValue(currentPoint.add(currentDirection.mapMultiply(a0)));
        mostRecentStepSizes[1] = mostRecentStepSizes[0];
        mostRecentStepSizes[0] = - phiPrime0 * Math.pow(a0, 2) / (2 * (phiA0 - phi0 - a0 * phiPrime0));

        // Ensure that we make reasonable progress and that the final step size is not too small
        if (Math.abs(mostRecentStepSizes[0] - mostRecentStepSizes[1]) <= MINIMUM_STEP_SIZE_CHANGE_THRESHOLD
                || mostRecentStepSizes[0] / mostRecentStepSizes[1] <= MINIMUM_STEP_SIZE_RATIO_THRESHOLD) {
            mostRecentStepSizes[0] = mostRecentStepSizes[1] / 2;
        }
     }

    private void performCubicInterpolation(RealVector currentPoint,
                                           RealVector currentDirection,
                                           double phi0,
                                           double phiPrime0) {
        double previousStepSize = mostRecentStepSizes[0];
        double a0 = mostRecentStepSizes[1];
        double a1 = mostRecentStepSizes[0];
        double a0Sq = Math.pow(a0, 2);
        double a1Sq = Math.pow(a1, 2);
        double a0Cub = Math.pow(a0, 3);
        double a1Cub = Math.pow(a1, 3);
        double phiA0 = objective.computeValue(currentPoint.add(currentDirection.mapMultiply(a0)));
        double phiA1 = objective.computeValue(currentPoint.add(currentDirection.mapMultiply(a1)));
        double denominator = a0Sq * a1Sq * (a1 - a0);
        double a = (a0Sq * (phiA1 - phi0 - a1 * phiPrime0) - a1Sq * (phiA0 - phi0 - a0 * phiPrime0)) / denominator;
        double b = (- a0Cub * (phiA1 - phi0 - a1 * phiPrime0) + a1Cub * (phiA0 - phi0 - a0 * phiPrime0)) / denominator;

        mostRecentStepSizes[0] = - (b - Math.sqrt(Math.pow(b, 2) - 3 * a * phiPrime0)) / (3 * a);
        mostRecentStepSizes[1] = previousStepSize;

        // Ensure that we make reasonable progress and that the final step size is not too small
        if (Math.abs(mostRecentStepSizes[0] - mostRecentStepSizes[1]) <= MINIMUM_STEP_SIZE_CHANGE_THRESHOLD
                || mostRecentStepSizes[0] / mostRecentStepSizes[1] <= MINIMUM_STEP_SIZE_RATIO_THRESHOLD) {
            mostRecentStepSizes[0] = mostRecentStepSizes[1] / 2;
        }
    }
}
