package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.platanios.learn.optimization.function.Function;
import org.platanios.learn.optimization.function.QuadraticFunction;
import org.platanios.learn.optimization.linesearch.*;

/**
 * @author Emmanouil Antonios Platanios
 */
abstract class AbstractNonlinearConjugateGradientSolver extends AbstractSolver {
    private final Function objective;

    /** Default value: If quadratic or linear function it is ExactLineSearch, otherwise it is StrongWolfeLineSearch
     * with CONSERVE_FIRST_ORDER_CHANGE for the step size initialization method. */
    private LineSearch lineSearch;

    // The following variables are used locally within iteration but are initialized here in order to make the code more
    // clear.
    double beta;

    public AbstractNonlinearConjugateGradientSolver(Function objective,
                                                    double[] initialPoint) {
        this.objective = objective;
        currentPoint = new ArrayRealVector(initialPoint);
        currentGradient = objective.computeGradient(currentPoint);
        currentDirection = currentGradient.mapMultiply(-1);
        currentObjectiveValue = objective.computeValue(currentPoint);
        currentIteration = 0;
        checkForPointConvergence = false;
        checkForObjectiveConvergence = false;

        if (objective instanceof QuadraticFunction) {
            this.lineSearch = new ExactLineSearch((QuadraticFunction) objective);
        } else {
            this.lineSearch = new StrongWolfeInterpolationLineSearch(
                    objective,
                    StepSizeInitializationMethod.CONSERVE_FIRST_ORDER_CHANGE,
                    1e-4,
                    0.25,
                    10
            );
        }
    }

    public void iterationUpdate() {
        previousPoint = currentPoint;
        previousGradient = currentGradient;
        previousDirection = currentDirection;
        previousStepSize = currentStepSize;
        previousObjectiveValue = currentObjectiveValue;
        currentStepSize = lineSearch.computeStepSize(currentPoint,
                                                     currentDirection,
                                                     previousPoint,
                                                     previousDirection,
                                                     previousStepSize);
        currentPoint = previousPoint.add(previousDirection.mapMultiply(currentStepSize));
        currentGradient = objective.computeGradient(currentPoint);
        beta = computeBeta();
        currentDirection = currentGradient.mapMultiply(-1).add(previousDirection.mapMultiply(beta));
        currentObjectiveValue = objective.computeValue(currentPoint);
    }

    public void printHeader() {
        System.out.println("Iteration #\tObjective Value\tPoint");
        System.out.println("===========\t===============\t=====");
    }

    public void printIteration() {
        System.out.format("%d\t\t\t%.10f\t%.5f\n", currentIteration, currentObjectiveValue, currentPoint.getEntry(0));
    }

    public abstract double computeBeta();

    public LineSearch getLineSearch() {
        return lineSearch;
    }

    public void setLineSearch(LineSearch lineSearch) {
        this.lineSearch = lineSearch;
    }
}
