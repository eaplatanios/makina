package org.platanios.learn.optimization.linesearch;

import com.google.common.base.Preconditions;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class IterativeLineSearch implements LineSearch {
    final Function objective;
    final StepSizeInitializationMethod stepSizeInitializationMethod;
    final double initialStepSize;

    double currentStepSize;

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

    public IterativeLineSearch(Function objective,
                               StepSizeInitializationMethod stepSizeInitializationMethod,
                               double initialStepSize) {
        Preconditions.checkArgument(initialStepSize > 0);

        this.objective = objective;
        this.stepSizeInitializationMethod = stepSizeInitializationMethod;

        if (stepSizeInitializationMethod == StepSizeInitializationMethod.CONSTANT) {
            this.initialStepSize = initialStepSize;
        } else {
            this.initialStepSize = 1;
            System.out.println("The initial step size argument is not needed " +
                    "for the selected step size initialization method!");
        }
    }

    public double computeStepSize(RealVector currentPoint,
                                  RealVector currentDirection) {
        switch (stepSizeInitializationMethod) {
            case UNIT:
                currentStepSize = 1.0;
                break;
            case CONSTANT:
                currentStepSize = initialStepSize;
                break;
            default:
                throw new IllegalArgumentException("More arguments are needed " +
                        "to compute the initial step size using the selected method!");
        }

        return performLineSearch(currentPoint, currentDirection);
    }

    public double computeStepSize(RealVector currentPoint,
                                  RealVector currentDirection,
                                  RealVector previousPoint,
                                  RealVector previousDirection) {
        double objectiveValueAtCurrentPoint = objective.computeValue(currentPoint);

        switch (stepSizeInitializationMethod) {
            case UNIT:
                currentStepSize = 1.0;
                System.out.println("Many of the passed arguments not needed in order " +
                        "to compute the initial step size using the selected method!");
                break;
            case CONSTANT:
                currentStepSize = initialStepSize;
                System.out.println("Many of the passed arguments not needed in order " +
                        "to compute the initial step size using the selected method!");
                break;
            case CONSERVE_FIRST_ORDER_CHANGE:
                throw new IllegalArgumentException("The previous step size is needed in order " +
                        "to compute the initial step size using the selected method!");
            case QUADRATIC_INTERPOLATION:
                if (previousDirection != null) {
                    currentStepSize = StepSizeInitialization.computeByQuadraticInterpolation(
                            objectiveValueAtCurrentPoint,
                            objective.computeValue(previousPoint),
                            objective.computeGradient(previousPoint),
                            previousDirection
                    );
                } else {
                    currentStepSize = 1.0;
                }
                break;
            case MODIFIED_QUADRATIC_INTERPOLATION:
                if (previousDirection != null) {
                    currentStepSize = StepSizeInitialization.computeByModifiedQuadraticInterpolation(
                            objectiveValueAtCurrentPoint,
                            objective.computeValue(previousPoint),
                            objective.computeGradient(previousPoint),
                            previousDirection
                    );
                } else {
                    currentStepSize = 1.0;
                }
                break;
            default:
                throw new NotImplementedException();
        }

        return performLineSearch(currentPoint, currentDirection);
    }

    public double computeStepSize(RealVector currentPoint,
                                     RealVector currentDirection,
                                     RealVector previousPoint,
                                     RealVector previousDirection,
                                     double previousStepSize) {
        double objectiveValueAtCurrentPoint = objective.computeValue(currentPoint);
        RealVector objectiveGradientAtCurrentPoint = objective.computeGradient(currentPoint);

        switch (stepSizeInitializationMethod) {
            case UNIT:
                currentStepSize = 1.0;
                System.out.println("Many of the passed arguments not needed in order " +
                        "to compute the initial step size using the selected method!");
                break;
            case CONSTANT:
                currentStepSize = initialStepSize;
                System.out.println("Many of the passed arguments not needed in order " +
                        "to compute the initial step size using the selected method!");
                break;
            case CONSERVE_FIRST_ORDER_CHANGE:
                if (previousDirection != null) {
                    currentStepSize = StepSizeInitialization.computeByConservingFirstOrderChange(
                            objectiveGradientAtCurrentPoint,
                            currentDirection,
                            objective.computeGradient(previousPoint),
                            previousDirection,
                            previousStepSize
                    );
                } else {
                    currentStepSize = 1.0;
                }
                break;
            case QUADRATIC_INTERPOLATION:
                if (previousDirection != null) {
                    currentStepSize = StepSizeInitialization.computeByQuadraticInterpolation(
                            objectiveValueAtCurrentPoint,
                            objective.computeValue(previousPoint),
                            objective.computeGradient(previousPoint),
                            previousDirection
                    );
                } else {
                    currentStepSize = 1.0;
                }
                System.out.println("The previous step size is not needed in order " +
                        "to compute the initial step size using the selected method!");
                break;
            case MODIFIED_QUADRATIC_INTERPOLATION:
                if (previousDirection != null) {
                    currentStepSize = StepSizeInitialization.computeByModifiedQuadraticInterpolation(
                            objectiveValueAtCurrentPoint,
                            objective.computeValue(previousPoint),
                            objective.computeGradient(previousPoint),
                            previousDirection
                    );
                } else {
                    currentStepSize = 1.0;
                }
                break;
            default:
                throw new NotImplementedException();
        }

        return performLineSearch(currentPoint, currentDirection);
    }

    public abstract double performLineSearch(RealVector currentPoint, RealVector direction);

    public StepSizeInitializationMethod getStepSizeInitializationMethod() {
        return stepSizeInitializationMethod;
    }
}
