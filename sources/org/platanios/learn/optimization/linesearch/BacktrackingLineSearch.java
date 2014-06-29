package org.platanios.learn.optimization.linesearch;

import com.google.common.base.Preconditions;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;

/**
 * Implements a simple backtracking line search algorithm. This algorithm starts from some initial step size value and
 * keeps reducing it, by multiplying it with a contraption factor, until it satisfies the Armijo condition (also known
 * as the sufficient decrease condition).
 *
 * @author Emmanouil Antonios Platanios
 */
public class BacktrackingLineSearch extends IterativeLineSearch {
    /** The contraption factor to use during the step size update on each failure to satisfy the Armijo condition. */
    private final double contraptionFactor;
    /** The proportionality constant to use for the Armijo condition. */
    private final double c;

    /**
     * Constructs a backtracking line search solver for the provided objective function instance and using the provided
     * step size initialization method to compute the initial step size. If the selected step size initialization method
     * is simply a constant value, then the alternative constructor must be used that receives the initial step size
     * value as its second argument.
     *
     * @param   objective                       The objective function instance.
     * @param   stepSizeInitializationMethod    The step size initialization method.
     * @param   contraptionFactor               The contraption factor to use during the step size update on each
     *                                          failure to satisfy the Armijo condition.
     * @param   c                               The proportionality constant to use for the Armijo condition.
     */
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

    /**
     * Constructs a backtracking line search solver for the provided objective function instance and using the provided
     * step size value as the initial step size. If another step size initialization method is required, then the
     * alternative constructor must be used that receives the step size initialization method as its second argument.
     *
     * @param   objective                       The objective function instance.
     * @param   contraptionFactor               The contraption factor to use during the step size update on each
     *                                          failure to satisfy the Armijo condition.
     * @param   c                               The proportionality constant to use for the Armijo condition.
     * @param   initialStepSize                 The initial step size value to use (it must have a value greater than
     *                                          zero).
     */
    public BacktrackingLineSearch(Function objective,
                                  double contraptionFactor,
                                  double c,
                                  double initialStepSize) {
        super(objective, initialStepSize);

        Preconditions.checkArgument(contraptionFactor > 0 && contraptionFactor < 1);
        Preconditions.checkArgument(c > 0 && c < 1);
        Preconditions.checkArgument(initialStepSize > 0);

        this.contraptionFactor = contraptionFactor;
        this.c = c;
    }

    /**
     * Constructs a backtracking line search solver for the provided objective function instance and using the provided
     * step size initialization method and the provided initial step size. Using this constructor only makes sense if
     * the selected step size initialization method is CONSTANT. For all other cases the extra initial step size
     * argument is not required. Furthermore, if the selected step size initialization method is not CONSTANT, then that
     * extra argument is completely ignored!
     *
     * @param   objective                       The objective function instance.
     * @param   stepSizeInitializationMethod    The step size initialization method.
     * @param   contraptionFactor               The contraption factor to use during the step size update on each
     *                                          failure to satisfy the Armijo condition.
     * @param   c                               The proportionality constant to use for the Armijo condition.
     * @param   initialStepSize                 The initial step size value to use (it must have a value greater than
     *                                          zero).
     */
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
     * {@inheritDoc}
     */
    @Override
    public double performLineSearch(RealVector point,
                                    RealVector direction) {
        double objectiveValueAtCurrentPoint = objective.computeValue(point);
        RealVector objectiveGradientAtCurrentPoint = objective.computeGradient(point);
        double currentStepSize = initialStepSize;

        while (!LineSearchConditions.checkArmijoCondition(objective,
                                                          point,
                                                          direction,
                                                          currentStepSize,
                                                          c,
                                                          objectiveValueAtCurrentPoint,
                                                          objectiveGradientAtCurrentPoint)) {
            currentStepSize *= contraptionFactor;
        }

        return currentStepSize;
    }
}
