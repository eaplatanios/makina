package org.platanios.learn.optimization.linesearch;

import com.google.common.base.Preconditions;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * Abstract class that all iterative line search algorithms should extend.
 *
 * @author Emmanouil Antonios Platanios
 */
abstract class IterativeLineSearch implements LineSearch {
    /** The objective function instance. */
    final Function objective;
    /** The step size initialization method. */
    final StepSizeInitializationMethod stepSizeInitializationMethod;

    /** The value of the initial step size (this value is set automatically using the chosen step size initialization
     * method). */
    double initialStepSize;

    /**
     * Constructs an iterative line search solver for the provided objective function instance and using the provided
     * step size initialization method to compute the initial step size. If the selected step size initialization method
     * is simply a constant value, then the alternative constructor must be used that receives the initial step size
     * value as its second argument.
     *
     * @param   objective                       The objective function instance.
     * @param   stepSizeInitializationMethod    The step size initialization method.
     */
    public IterativeLineSearch(Function objective,
                               StepSizeInitializationMethod stepSizeInitializationMethod) {
        this.objective = objective;
        this.stepSizeInitializationMethod = stepSizeInitializationMethod;
        this.initialStepSize = 1;

        if (stepSizeInitializationMethod == StepSizeInitializationMethod.CONSTANT) {
            throw new IllegalArgumentException("An initial step size value is required " +
                    "when the step size initialization method is set to CONSTANT!");
        }
    }

    /**
     * Constructs an iterative line search solver for the provided objective function instance and using the provided
     * step size value as the initial step size. If another step size initialization method is required, then the
     * alternative constructor must be used that receives the step size initialization method as its second argument.
     *
     * @param   objective           The objective function instance.
     * @param   initialStepSize     The initial step size value to use (it must have a value greater than zero).
     */
    public IterativeLineSearch(Function objective,
                               double initialStepSize) {
        Preconditions.checkArgument(initialStepSize > 0);

        this.objective = objective;
        this.stepSizeInitializationMethod = StepSizeInitializationMethod.CONSTANT;
        this.initialStepSize = initialStepSize;
    }

    /**
     * Constructs an iterative line search solver for the provided objective function instance and using the provided
     * step size initialization method and the provided initial step size. Using this constructor only makes sense if
     * the selected step size initialization method is CONSTANT. For all other cases the extra initial step size
     * argument is not required. Furthermore, if the selected step size initialization method is not CONSTANT, then that
     * extra argument is completely ignored!
     *
     * @param   objective                       The objective function instance.
     * @param   stepSizeInitializationMethod    The step size initialization method.
     * @param   initialStepSize                 The initial step size value to use (it must have a value greater than
     *                                          zero).
     */
    public IterativeLineSearch(Function objective,
                               StepSizeInitializationMethod stepSizeInitializationMethod,
                               double initialStepSize) {
        Preconditions.checkArgument(initialStepSize > 0);

        this.objective = objective;
        this.stepSizeInitializationMethod = stepSizeInitializationMethod;

        switch (stepSizeInitializationMethod) {
            case UNIT:
                this.initialStepSize = 1;
                if (initialStepSize != 1) {
                    System.err.println("WARNING: The selected step size initialization method is UNIT, but the " +
                            "selected initial step size is not 1.0! A value of 1.0 will be used for the initial " +
                            "step size!");
                } else {
                    System.out.println("An initial step size value is not required when the selected step size " +
                            "initialization method is UNIT.");
                }
                break;
            case CONSTANT:
                this.initialStepSize = initialStepSize;
                if (initialStepSize == 1) {
                    System.out.println("The selected step size initialization method is CONSTANT and the initial" +
                            "step size value is 1. That is equivalent to simply using UNIT as the selected step size" +
                            "initialization method.");
                }
                break;
            default:
                System.out.println("An initial step size value is not required when the selected step size " +
                        "initialization method is not CONSTANT.");
        }
    }

    /**
     * {@inheritDoc}
     */
    public double computeStepSize(RealVector point,
                                  RealVector direction,
                                  RealVector previousPoint,
                                  RealVector previousDirection,
                                  double previousStepSize) {
        switch (stepSizeInitializationMethod) {
            case UNIT:      // Initial step size for this case was set in the constructor.
            case CONSTANT:  // Initial step size for this case was set in the constructor.
                break;
            case CONSERVE_FIRST_ORDER_CHANGE:
                // Check whether the previous direction is set (if it is not it means that we are on the first iteration
                // of the optimization algorithm).
                if (previousDirection != null) {
                    initialStepSize = StepSizeInitialization.computeByConservingFirstOrderChange(
                            objective.computeGradient(point),
                            direction,
                            objective.computeGradient(previousPoint),
                            previousDirection,
                            previousStepSize
                    );
                } else {
                    initialStepSize = 1.0;
                }
                break;
            case QUADRATIC_INTERPOLATION:
                // Check whether the previous direction is set (if it is not it means that we are on the first iteration
                // of the optimization algorithm).
                if (previousDirection != null) {
                    initialStepSize = StepSizeInitialization.computeByQuadraticInterpolation(
                            objective.computeValue(point),
                            objective.computeValue(previousPoint),
                            objective.computeGradient(previousPoint),
                            previousDirection
                    );
                } else {
                    initialStepSize = 1.0;
                }
                break;
            case MODIFIED_QUADRATIC_INTERPOLATION:
                // Check whether the previous direction is set (if it is not it means that we are on the first iteration
                // of the optimization algorithm).
                if (previousDirection != null) {
                    initialStepSize = StepSizeInitialization.computeByModifiedQuadraticInterpolation(
                            objective.computeValue(point),
                            objective.computeValue(previousPoint),
                            objective.computeGradient(previousPoint),
                            previousDirection
                    );
                } else {
                    initialStepSize = 1.0;
                }
                break;
            default:
                throw new NotImplementedException();
        }

        return performLineSearch(point, direction);
    }

    /**
     * Perform the actual line search after an initial value for the step size has been selected. The
     * {@code initialStepSize} must have been initialized before this method is called.
     *
     * @param   point       The point at which we perform the line search.
     * @param   direction   The direction for which we perform the line search.
     * @return              A step size value that satisfies certain criteria that depend on the algorithm choice.
     */
    public abstract double performLineSearch(RealVector point, RealVector direction);
}
