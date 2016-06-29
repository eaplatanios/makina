package makina.optimization.linesearch;

import com.google.common.base.Preconditions;
import makina.math.matrix.Vector;
import makina.optimization.function.NonSmoothFunctionException;
import makina.optimization.function.AbstractFunction;

/**
 * Implements a simple backtracking line search algorithm. This algorithm starts from some initial step size value and
 * keeps reducing it, by multiplying it with a contraption factor, until it satisfies the Armijo condition (also known
 * as the sufficient decrease condition).
 *
 * @author Emmanouil Antonios Platanios
 */
public final class BacktrackingLineSearch extends IterativeLineSearch {
    /** The contraption factor to use during the step size update on each failure to satisfy the Armijo condition. */
    private final double contraptionFactor;
    /** The proportionality constant to use for the Armijo condition. */
    private final double c;
    /** Tolerance for the step size value. That is, if during the search the step size value if smaller than this
     * tolerance, it is returned. */
    private final double tolerance;

    /**
     * Constructs a backtracking line search solver for the provided objective function instance and using the provided
     * parameters. Note that in this case a default tolerance value of 1e-10 is used.
     *
     * @param   objective           The objective function instance.
     * @param   contraptionFactor   The contraption factor to use during the step size update on each failure to satisfy
     *                              the Armijo condition. The value provided must lie between 0 and 1.
     * @param   c                   The proportionality constant to use for the Armijo condition. The value provided
     *                              must lie between 0 and 1.
     */
    public BacktrackingLineSearch(AbstractFunction objective, double contraptionFactor, double c) {
        this(objective, contraptionFactor, c, 1e-10);
    }

    /**
     * Constructs a backtracking line search solver for the provided objective function instance and using the provided
     * parameters and the provided tolerance for the step size value.
     *
     * @param   objective           The objective function instance.
     * @param   contraptionFactor   The contraption factor to use during the step size update on each failure to satisfy
     *                              the Armijo condition. The value provided must lie between 0 and 1.
     * @param   c                   The proportionality constant to use for the Armijo condition. The value provided
     *                              must lie between 0 and 1.
     * @param   tolerance           Tolerance for the step size value. That is, if during the search the step size value
     *                              is smaller than this tolerance, then it is immediately returned.
     */
    public BacktrackingLineSearch(AbstractFunction objective, double contraptionFactor, double c, double tolerance) {
        super(objective);

        Preconditions.checkArgument(contraptionFactor > 0 && contraptionFactor < 1);
        Preconditions.checkArgument(c > 0 && c < 1);

        this.contraptionFactor = contraptionFactor;
        this.c = c;
        this.tolerance = tolerance;
    }

    /** {@inheritDoc} */
    @Override
    public double performLineSearch(Vector point,
                                    Vector direction)
            throws NonSmoothFunctionException {
        double objectiveValueAtCurrentPoint = objective.getValue(point);
        Vector objectiveGradientAtCurrentPoint = objective.getGradient(point);
        double currentStepSize = initialStepSize;

        while (!LineSearchConditions.checkArmijoCondition(objective,
                                                          point,
                                                          direction,
                                                          currentStepSize,
                                                          c,
                                                          objectiveValueAtCurrentPoint,
                                                          objectiveGradientAtCurrentPoint)) {
            currentStepSize *= contraptionFactor;
            if (currentStepSize < tolerance)
                return currentStepSize;
        }

        return currentStepSize;
    }
}
