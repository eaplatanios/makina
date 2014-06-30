package org.platanios.learn.optimization.linesearch;

import com.google.common.base.Preconditions;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.AbstractFunction;

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
     * parameters.
     *
     * @param   objective           The objective function instance.
     * @param   contraptionFactor   The contraption factor to use during the step size update on each failure to satisfy
     *                              the Armijo condition. The value provided must lie between 0 and 1.
     * @param   c                   The proportionality constant to use for the Armijo condition. The value provided
     *                              must lie between 0 and 1.
     */
    public BacktrackingLineSearch(AbstractFunction objective, double contraptionFactor, double c) {
        super(objective);

        Preconditions.checkArgument(contraptionFactor > 0 && contraptionFactor < 1);
        Preconditions.checkArgument(c > 0 && c < 1);

        this.contraptionFactor = contraptionFactor;
        this.c = c;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double performLineSearch(RealVector point,
                                    RealVector direction) {
        double objectiveValueAtCurrentPoint = objective.getValue(point);
        RealVector objectiveGradientAtCurrentPoint = objective.getGradient(point);
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
