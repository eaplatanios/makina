package org.platanios.optimization.linesearch;

import com.google.common.base.Preconditions;
import org.platanios.math.matrix.Vector;
import org.platanios.optimization.function.AbstractFunction;
import org.platanios.optimization.function.NonSmoothFunctionException;

/**
 * Abstract class that all iterative line search algorithms should extend.
 *
 * @author Emmanouil Antonios Platanios
 */
abstract class IterativeLineSearch implements LineSearch {
    /** The objective function instance. */
    final AbstractFunction objective;

    /** The step size initialization method. */
    StepSizeInitializationMethod stepSizeInitializationMethod = StepSizeInitializationMethod.UNIT;
    /** The value of the initial step size (this value is set automatically using the chosen step size initialization
     * method). */
    double initialStepSize = 1;

    /**
     * Constructs an iterative line search solver for the provided objective function instance.
     *
     * @param   objective   The objective function instance.
     */
    public IterativeLineSearch(AbstractFunction objective) {
        this.objective = objective;
    }

    /** {@inheritDoc} */
    @Override
    public double computeStepSize(int iterationNumber,
                                  Vector point,
                                  Vector direction,
                                  Vector previousPoint,
                                  Vector previousDirection,
                                  double previousStepSize)
            throws NonSmoothFunctionException {
        initialStepSize = stepSizeInitializationMethod.computeInitialStepSize(objective,
                                                                              point,
                                                                              direction,
                                                                              previousPoint,
                                                                              previousDirection,
                                                                              initialStepSize,
                                                                              previousStepSize);
        return performLineSearch(point, direction);
    }

    /**
     * Perform the actual line search after an initial value for the step size has been selected. The
     * {@link #initialStepSize} must have been initialized before this method is called.
     *
     * @param   point       The point at which we perform the line search.
     * @param   direction   The direction for which we perform the line search.
     * @return              A step size value that satisfies certain criteria that depend on the algorithm choice.
     */
    public abstract double performLineSearch(Vector point, Vector direction) throws NonSmoothFunctionException;

    /**
     * Gets the step size initialization method used by this iterative line search instance.
     *
     * @return  The step size initialization method used by this iterative line search instance.
     */
    public StepSizeInitializationMethod getStepSizeInitializationMethod() {
        return stepSizeInitializationMethod;
    }

    /**
     * Sets the step size initialization method for this iterative line search instance. If the selected method is
     * {@link StepSizeInitializationMethod#CONSTANT} and the desired
     * initial step size value to be used is not 1, then {@link #setInitialStepSize(double)} must be called as well to
     * set the constant value of the initial step size.
     *
     * @param   stepSizeInitializationMethod    The step size initialization method to be used.
     */
    public void setStepSizeInitializationMethod(StepSizeInitializationMethod stepSizeInitializationMethod) {
        this.stepSizeInitializationMethod = stepSizeInitializationMethod;
    }

    /**
     * Gets the initial step size value used by this iterative line search instance. If the selected step size
     * initialization method is
     * {@link StepSizeInitializationMethod#CONSTANT}, then the initial step
     * value can have any positive real number value. Otherwise, the initial step size value that is returned by this
     * method is not actually being used by this iterative line search instance. Note that if the step size
     * initialization method is {@link StepSizeInitializationMethod#UNIT},
     * then the initial step size value that is always used by this iterative line search instance is 1.
     *
     * @return  The value of the initial step size.
     */
    public double getInitialStepSize() {
        return initialStepSize;
    }

    /**
     * Sets the initial step size value used by this iterative line search instance. This method is only useful if the
     * selected step size initialization method is
     * {@link StepSizeInitializationMethod#CONSTANT}. Otherwise, the
     * initial step size variable is not actually being used by this iterative line search instance. Note that if the
     * step size initialization method is
     * {@link StepSizeInitializationMethod#UNIT}, then the initial step
     * size value that is always used by this iterative line search instance is 1.
     *
     * @param   initialStepSize     The initial step size value to use. The value provided must be a positive real
     *                              number.
     */
    public void setInitialStepSize(double initialStepSize) {
        Preconditions.checkArgument(initialStepSize > 0);
        this.initialStepSize = initialStepSize;
    }
}
