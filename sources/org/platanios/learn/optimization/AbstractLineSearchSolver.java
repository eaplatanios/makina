package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;
import org.platanios.learn.optimization.function.QuadraticFunction;
import org.platanios.learn.optimization.linesearch.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractLineSearchSolver extends AbstractSolver {
    final Function objective;

    /** Default value: If quadratic or linear function it is ExactLineSearch, otherwise it is StrongWolfeLineSearch
     * with CONSERVE_FIRST_ORDER_CHANGE for the step size initialization method. */
    private LineSearch lineSearch;

    RealVector currentDirection;
    RealVector previousDirection;
    double currentStepSize;
    double previousStepSize;
    RealVector previousPoint;
    double currentObjectiveValue;
    double previousObjectiveValue;
    double pointL2NormChange;
    double objectiveChange;
    double pointL2NormChangeTolerance = 1e-10;
    double objectiveChangeTolerance = 1e-10;
    boolean pointL2NormConverged = false;
    boolean gradientConverged = false;
    boolean objectiveConverged = false;

    public AbstractLineSearchSolver(Function objective,
                                    double[] initialPoint) {
        this.objective = objective;
        this.currentPoint = new ArrayRealVector(initialPoint);
        currentObjectiveValue = objective.computeValue(currentPoint);
        currentIteration = 0;

        if (objective.getClass() == QuadraticFunction.class) {
            this.lineSearch = new ExactLineSearch((QuadraticFunction) objective);
        } else {
            this.lineSearch = new StrongWolfeLineSearch(
                    objective,
                    StepSizeInitializationMethod.CONSERVE_FIRST_ORDER_CHANGE,
                    1e-4,
                    0.9,
                    10
            );
        }
    }
    public boolean checkTerminationConditions() {
        if (currentIteration > 0) {
            pointL2NormChange = currentPoint.subtract(previousPoint).getNorm();
            objectiveChange = Math.abs((previousObjectiveValue - currentObjectiveValue) / previousObjectiveValue);
            pointL2NormConverged = pointL2NormChange <= pointL2NormChangeTolerance;
            objectiveConverged = objectiveChange <= objectiveChangeTolerance;
            gradientConverged = objective.computeGradient(currentPoint).getNorm() == 0.0;

            return pointL2NormConverged || objectiveConverged || gradientConverged;
        } else {
            return false;
        }
    }

    public void iterationUpdate() {
        updateDirection();
        updateStepSize();
        previousDirection = currentDirection;
        previousStepSize = currentStepSize;
        previousPoint = currentPoint;
        previousObjectiveValue = currentObjectiveValue;
        updatePoint();
        currentObjectiveValue = objective.computeValue(currentPoint);
    }

    public void printHeader() {
        System.out.println("Iteration #\tObjective Value\tPoint");
        System.out.println("===========\t===============\t=====");
    }

    public void printIteration() {
        System.out.format("%d\t\t\t%.10f\t%.5f\n", currentIteration, currentObjectiveValue, currentPoint.getEntry(0));
    }

    public void printTerminationMessage() {
        if (pointL2NormConverged) {
            System.out.println("The L2 norm of the point change, "
                                       + DECIMAL_FORMAT.format(pointL2NormChange)
                                       + ", was below the convergence threshold of "
                                       + DECIMAL_FORMAT.format(pointL2NormChangeTolerance)
                                       + "!\n");
        }
        if (objectiveConverged) {
            System.out.println("The relative change of the objective function value, "
                                       + DECIMAL_FORMAT.format(objectiveChange)
                                       + ", was below the convergence threshold of "
                                       + DECIMAL_FORMAT.format(objectiveChangeTolerance)
                                       + "!\n");
        }
        if (gradientConverged) {
            System.out.println("The gradient became 0!\n");
        }
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

    public void setLineSearch(LineSearch lineSearch) {
        this.lineSearch = lineSearch;
    }
}
