package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.QuadraticFunction;
import org.platanios.learn.optimization.linesearch.*;

/**
 * @author Emmanouil Antonios Platanios
 */
abstract class AbstractLineSearchSolver extends AbstractSolver {
    /** Default value: If quadratic or linear function it is ExactLineSearch, otherwise it is StrongWolfeLineSearch
     * with CONSERVE_FIRST_ORDER_CHANGE for the step size initialization method. */
    private LineSearch lineSearch;

    public AbstractLineSearchSolver(AbstractFunction objective,
                                    double[] initialPoint) {
        this.objective = objective;
        this.currentPoint = new ArrayRealVector(initialPoint);
        currentObjectiveValue = objective.getValue(currentPoint);
        currentGradient = objective.getGradient(currentPoint);
        currentIteration = 0;

        if (objective instanceof QuadraticFunction) {
            this.lineSearch = new ExactLineSearch((QuadraticFunction) objective);
        } else {
            this.lineSearch = new StrongWolfeInterpolationLineSearch(
                    objective,
                    StepSizeInitializationMethod.CONSERVE_FIRST_ORDER_CHANGE,
                    1e-4,
                    0.9,
                    10
            );
        }
    }

    @Override
    public void iterationUpdate() {
        previousGradient = currentGradient;
        previousDirection = currentDirection;
        updateDirection();
        previousStepSize = currentStepSize;
        updateStepSize();
        previousPoint = currentPoint;
        previousObjectiveValue = currentObjectiveValue;
        updatePoint();
        currentObjectiveValue = objective.getValue(currentPoint);
    }

    public void updateStepSize() {
        currentStepSize = lineSearch.computeStepSize(currentPoint,
                                                     currentDirection,
                                                     previousPoint,
                                                     previousDirection,
                                                     previousStepSize);
    }

    public abstract void updateDirection();
    public abstract void updatePoint();

    public LineSearch getLineSearch() {
        return lineSearch;
    }

    public void setLineSearch(LineSearch lineSearch) {
        this.lineSearch = lineSearch;
    }
}
