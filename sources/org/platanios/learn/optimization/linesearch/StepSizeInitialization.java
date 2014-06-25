package org.platanios.learn.optimization.linesearch;

import org.apache.commons.math3.linear.RealVector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class StepSizeInitialization {
    public static double computeByConservingFirstOrderChange(RealVector objectiveGradientAtCurrentPoint,
                                                             RealVector currentDirection,
                                                             RealVector objectiveGradientAtPreviousPoint,
                                                             RealVector previousDirection,
                                                             double previousStepSize) {
        return previousStepSize
                * objectiveGradientAtPreviousPoint.dotProduct(previousDirection)
                /  objectiveGradientAtCurrentPoint.dotProduct(currentDirection);
    }

    public static double computeByQuadraticInterpolation(double objectiveValueAtCurrentPoint,
                                                         double objectiveValueAtPreviousPoint,
                                                         RealVector objectiveGradientAtPreviousPoint,
                                                         RealVector previousDirection) {
        return 2 * (objectiveValueAtCurrentPoint - objectiveValueAtPreviousPoint)
                / objectiveGradientAtPreviousPoint.dotProduct(previousDirection);
    }

    public static double computeByModifiedQuadraticInterpolation(double objectiveValueAtCurrentPoint,
                                                                 double objectiveValueAtPreviousPoint,
                                                                 RealVector objectiveGradientAtPreviousPoint,
                                                                 RealVector previousDirection) {
        return Math.min(1, 2.02 * (objectiveValueAtCurrentPoint - objectiveValueAtPreviousPoint)
                / objectiveGradientAtPreviousPoint.dotProduct(previousDirection));
    }
}
