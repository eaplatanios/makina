package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;
import org.platanios.learn.optimization.function.QuadraticFunction;
import org.platanios.learn.optimization.linesearch.*;

/**
 * @author Emmanouil Antonios Platanios
 */
abstract class AbstractLineSearchSolver extends AbstractSolver {
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

    private double pointChangeTolerance = 1e-10;
    private double objectiveChangeTolerance = 1e-10;
    private double gradientTolerance = 1e-10;

    private double pointChange;
    private double objectiveChange;
    private double gradientNorm;

    private boolean pointConverged = false;
    private boolean gradientConverged = false;
    private boolean objectiveConverged = false;

    public AbstractLineSearchSolver(Function objective,
                                    double[] initialPoint) {
        this.objective = objective;
        this.currentPoint = new ArrayRealVector(initialPoint);
        currentObjectiveValue = objective.computeValue(currentPoint);
        currentIteration = 0;

        if (objective instanceof QuadraticFunction) {
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
            pointChange = currentPoint.subtract(previousPoint).getNorm();
            objectiveChange = Math.abs((previousObjectiveValue - currentObjectiveValue) / previousObjectiveValue);
            gradientNorm = objective.computeGradient(currentPoint).getNorm();

            pointConverged = pointChange <= pointChangeTolerance;
            objectiveConverged = objectiveChange <= objectiveChangeTolerance;
            gradientConverged = gradientNorm <= gradientTolerance;

            return pointConverged || objectiveConverged || gradientConverged;
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
        if (pointConverged) {
            System.out.println("The L2 norm of the point change, "
                                       + DECIMAL_FORMAT.format(pointChange)
                                       + ", was below the convergence threshold of "
                                       + DECIMAL_FORMAT.format(pointChangeTolerance)
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
            System.out.println("The gradient norm became "
                                       + DECIMAL_FORMAT.format(gradientNorm)
                                       + ", which is less than the convergence threshold of "
                                       + DECIMAL_FORMAT.format(gradientTolerance)
                                       + "!\n");
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

    public LineSearch getLineSearch() {
        return lineSearch;
    }

    public void setLineSearch(LineSearch lineSearch) {
        this.lineSearch = lineSearch;
    }

    public double getPointChangeTolerance() {
        return pointChangeTolerance;
    }

    public void setPointChangeTolerance(double pointChangeTolerance) {
        this.pointChangeTolerance = pointChangeTolerance;
    }

    public double getObjectiveChangeTolerance() {
        return objectiveChangeTolerance;
    }

    public void setObjectiveChangeTolerance(double objectiveChangeTolerance) {
        this.objectiveChangeTolerance = objectiveChangeTolerance;
    }

    public double getGradientTolerance() {
        return gradientTolerance;
    }

    public void setGradientTolerance(double gradientTolerance) {
        this.gradientTolerance = gradientTolerance;
    }
}
