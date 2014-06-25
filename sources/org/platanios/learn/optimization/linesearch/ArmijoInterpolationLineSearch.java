package org.platanios.learn.optimization.linesearch;

import com.google.common.base.Preconditions;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ArmijoInterpolationLineSearch extends IterativeLineSearch {
    private final double c;

    private double[] mostRecentStepSizes; // [0] is the previous one and [1] is the one before that one

    public ArmijoInterpolationLineSearch(Function objectiveFunction,
                                         StepSizeInitializationMethod stepSizeInitializationMethod,
                                         double c) {
        super(objectiveFunction, stepSizeInitializationMethod);
        Preconditions.checkArgument(c > 0 && c < 1);
        this.c = c;
        mostRecentStepSizes = new double[2];
    }

    public ArmijoInterpolationLineSearch(Function objectiveFunction,
                                         StepSizeInitializationMethod stepSizeInitializationMethod,
                                         double c,
                                         double initialStepSize) {
        super(objectiveFunction, stepSizeInitializationMethod, initialStepSize);
        Preconditions.checkArgument(c > 0 && c < 1);
        Preconditions.checkArgument(initialStepSize > 0);
        this.c = c;
        mostRecentStepSizes = new double[2];
    }

    public void performLineSearch(RealVector currentPoint,
                                   RealVector currentDirection) {
        double objectiveValueAtCurrentPoint = objectiveFunction.computeValue(currentPoint);
        RealVector objectiveGradientAtCurrentPoint = objectiveFunction.computeGradient(currentPoint);
        double dotProductOfObjectiveGradientAndDirection = objectiveGradientAtCurrentPoint.dotProduct(currentDirection);

        mostRecentStepSizes[0] = currentStepSize;
        int iterationNumber = 0;

        while (!LineSearchConditions.checkArmijoCondition(objectiveFunction,
                currentPoint,
                currentDirection,
                mostRecentStepSizes[0],
                c,
                objectiveValueAtCurrentPoint,
                objectiveGradientAtCurrentPoint)) {
            if (iterationNumber == 0) {
                performQuadraticInterpolation(
                        currentPoint,
                        currentDirection,
                        objectiveValueAtCurrentPoint,
                        dotProductOfObjectiveGradientAndDirection
                );
            } else {
                performCubicInterpolation(
                        currentPoint,
                        currentDirection,
                        objectiveValueAtCurrentPoint,
                        dotProductOfObjectiveGradientAndDirection
                );
            }
            iterationNumber++;
            if (iterationNumber >= 10) {
                break;
            }
        }

        currentStepSize = mostRecentStepSizes[0];
    }

    private void performQuadraticInterpolation(RealVector currentPoint,
                                               RealVector currentDirection,
                                               double objectiveValueAtCurrentPoint,
                                               double dotProductOfObjectiveGradientAndDirection) {
        RealVector pointWithA0 = currentPoint.add(currentDirection.mapMultiply(mostRecentStepSizes[0]));
        double phiA0 = objectiveFunction.computeValue(pointWithA0);
        mostRecentStepSizes[1] = mostRecentStepSizes[0];
        mostRecentStepSizes[0] = - dotProductOfObjectiveGradientAndDirection
                * Math.pow(mostRecentStepSizes[1], 2)
                / (2 * (phiA0
                    - objectiveValueAtCurrentPoint
                    - mostRecentStepSizes[1] * dotProductOfObjectiveGradientAndDirection));
     }

    private void performCubicInterpolation(RealVector currentPoint,
                                           RealVector currentDirection,
                                           double objectiveValueAtCurrentPoint,
                                           double dotProductOfObjectiveGradientAndDirection) {
        double previousStepSize = mostRecentStepSizes[0];
        double a0 = mostRecentStepSizes[1];
        double a1 = mostRecentStepSizes[0];
        double a0_squared = Math.pow(a0, 2);
        double a1_squared = Math.pow(a1, 2);
        double a0_cubed = Math.pow(a0, 3);
        double a1_cubed = Math.pow(a1, 3);
        RealVector pointWithA0 = currentPoint.add(currentDirection.mapMultiply(a0));
        RealVector pointWithA1 = currentPoint.add(currentDirection.mapMultiply(a1));
        double phiA0 = objectiveFunction.computeValue(pointWithA0);
        double phiA1 = objectiveFunction.computeValue(pointWithA1);

        double denominator = a0_squared * a1_squared * (a1 - a0);
        double a = (a0_squared * (phiA1 - objectiveValueAtCurrentPoint - a1 * dotProductOfObjectiveGradientAndDirection)
                - a1_squared * (phiA0 - objectiveValueAtCurrentPoint - a0 * dotProductOfObjectiveGradientAndDirection))
                / denominator;
        double b = (- a0_cubed * (phiA1 - objectiveValueAtCurrentPoint - a1 * dotProductOfObjectiveGradientAndDirection)
                + a1_cubed * (phiA0 - objectiveValueAtCurrentPoint - a0 * dotProductOfObjectiveGradientAndDirection))
                / denominator;

        mostRecentStepSizes[0] = - (b - Math.sqrt(Math.pow(b, 2) - 3 * a * dotProductOfObjectiveGradientAndDirection))
                / (3 * a);
        mostRecentStepSizes[1] = previousStepSize;

        if (Math.abs(mostRecentStepSizes[0] - mostRecentStepSizes[1]) <= 1e-3
                || mostRecentStepSizes[0] / mostRecentStepSizes[1] <= 1e-1) {
            mostRecentStepSizes[0] = mostRecentStepSizes[1] / 2;
        }
    }
}
