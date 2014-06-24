package org.platanios.learn.optimization;

import com.google.common.base.Preconditions;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;

/**
 * @author Emmanouil Antonios Platanios
 */
public class BacktrackingLineSearch implements LineSearch {
    private final Function objectiveFunction;
    private final double initialStepSize;
    private final double contraptionFactor;
    private final double c;

    public BacktrackingLineSearch(Function objectiveFunction,
                                  double initialStepSize,
                                  double contraptionFactor,
                                  double c) {
        Preconditions.checkArgument(initialStepSize > 0);
        Preconditions.checkArgument(contraptionFactor > 0 && contraptionFactor < 1);
        Preconditions.checkArgument(c > 0 && c < 1);

        this.objectiveFunction = objectiveFunction;
        this.initialStepSize = initialStepSize;
        this.contraptionFactor = contraptionFactor;
        this.c = c;
    }

    /**
     * Computes the step size value using the backtracking line search algorithm. {@code initialStepSize} should be
     * chosen to be {@code 1} in Newton and quasi-Newton methods, but can have different values in other algorithms,
     * such as steepest descent or conjugate gradient. The method employed for stopping the line search in this
     * algorithm is well suited for Newton methods, but is less appropriate for quasi-Newton and conjugate gradient
     * methods.
     *
     * @param   currentPoint
     * @param   direction
     * @return
     */
    public double computeStepSize(RealVector currentPoint,
                                  RealVector direction) {
        double objectiveFunctionValueAtCurrentPoint = objectiveFunction.computeValue(currentPoint);
        RealVector objectiveFunctionGradientAtCurrentPoint = objectiveFunction.computeGradient(currentPoint);

        double currentStepSize = initialStepSize;

        while (!LineSearchConditions.checkArmijoCondition(objectiveFunction,
                                                          currentPoint,
                                                          direction,
                                                          currentStepSize,
                                                          c,
                                                          objectiveFunctionValueAtCurrentPoint,
                                                          objectiveFunctionGradientAtCurrentPoint)) {
            currentStepSize *= contraptionFactor;
        }

        return currentStepSize;
    }
}
