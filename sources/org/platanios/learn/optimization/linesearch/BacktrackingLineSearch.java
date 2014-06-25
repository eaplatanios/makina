package org.platanios.learn.optimization.linesearch;

import com.google.common.base.Preconditions;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;

/**
 * @author Emmanouil Antonios Platanios
 */
public class BacktrackingLineSearch extends IterativeLineSearch {
    private final double contraptionFactor;
    private final double c;

    public BacktrackingLineSearch(Function objective,
                                  StepSizeInitializationMethod stepSizeInitializationMethod,
                                  double contraptionFactor,
                                  double c) {
        super(objective, stepSizeInitializationMethod);

        Preconditions.checkArgument(contraptionFactor > 0 && contraptionFactor < 1);
        Preconditions.checkArgument(c > 0 && c < 1);

        this.contraptionFactor = contraptionFactor;
        this.c = c;
    }

    public BacktrackingLineSearch(Function objective,
                                  StepSizeInitializationMethod stepSizeInitializationMethod,
                                  double contraptionFactor,
                                  double c,
                                  double initialStepSize) {
        super(objective, stepSizeInitializationMethod, initialStepSize);

        Preconditions.checkArgument(contraptionFactor > 0 && contraptionFactor < 1);
        Preconditions.checkArgument(c > 0 && c < 1);
        Preconditions.checkArgument(initialStepSize > 0);

        this.contraptionFactor = contraptionFactor;
        this.c = c;
    }

    /**
     * Computes the step size value using the backtracking line search algorithm. {@code INITIAL_STEP_SIZE} should be
     * chosen to be {@code 1} in Newton and quasi-Newton methods, but can have different values in other algorithms,
     * such as steepest descent or conjugate gradient. The method employed for stopping the line search in this
     * algorithm is well suited for Newton methods, but is less appropriate for quasi-Newton and conjugate gradient
     * methods.
     *
     * @param   currentPoint
     * @param   currentDirection
     * @return
     */
    public double performLineSearch(RealVector currentPoint,
                                    RealVector currentDirection) {
        double objectiveValueAtCurrentPoint = objective.computeValue(currentPoint);
        RealVector objectiveGradientAtCurrentPoint = objective.computeGradient(currentPoint);

        while (!LineSearchConditions.checkArmijoCondition(objective,
                                                          currentPoint,
                                                          currentDirection,
                                                          currentStepSize,
                c,
                                                          objectiveValueAtCurrentPoint,
                                                          objectiveGradientAtCurrentPoint)) {
            currentStepSize *= contraptionFactor;
        }

        return currentStepSize;
    }
}
