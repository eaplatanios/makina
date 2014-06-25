package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;
import org.platanios.learn.optimization.function.QuadraticFunction;
import org.platanios.learn.optimization.linesearch.*;

import java.text.DecimalFormat;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractSolver implements Solver {
    final Function objective;
    static final DecimalFormat DECIMAL_FORMAT = new DecimalFormat("0.##E0");

    /** Default value: If quadratic or linear function it is ExactLineSearch, otherwise it is StrongWolfeLineSearch
     * with CONSERVE_FIRST_ORDER_CHANGE for the step size initialization method. */
    private LineSearch lineSearch;

    int currentIteration;
    RealVector currentDirection;
    RealVector previousDirection;
    double currentStepSize;
    double previousStepSize;
    RealVector currentPoint;
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

    public AbstractSolver(Function objective,
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

    public void updateStepSize() {
        if (!(lineSearch instanceof IterativeLineSearch)) {
            currentStepSize = lineSearch.computeStepSize(currentPoint, currentDirection);
            return;
        }

        switch (((IterativeLineSearch) this.lineSearch).getStepSizeInitializationMethod()) {
            case UNIT:
            case CONSTANT:
                currentStepSize = lineSearch.computeStepSize(currentPoint, currentDirection);
                break;
            case CONSERVE_FIRST_ORDER_CHANGE:
                currentStepSize = ((IterativeLineSearch) lineSearch).computeStepSize(
                        currentPoint,
                        currentDirection,
                        previousPoint,
                        previousDirection,
                        previousStepSize
                );
                break;
            case QUADRATIC_INTERPOLATION:
                currentStepSize = ((IterativeLineSearch) lineSearch).computeStepSize(
                        currentPoint,
                        currentDirection,
                        previousPoint,
                        previousDirection
                );
                break;
        }
    }

    public boolean checkForConvergence() {
        pointL2NormChange = currentPoint.subtract(previousPoint).getNorm();
        objectiveChange = Math.abs((previousObjectiveValue - currentObjectiveValue) / previousObjectiveValue);
        pointL2NormConverged = pointL2NormChange <= pointL2NormChangeTolerance;
        objectiveConverged = objectiveChange <= objectiveChangeTolerance;
        gradientConverged = objective.computeGradient(currentPoint).getNorm() == 0.0;

        return pointL2NormConverged || objectiveConverged || gradientConverged;
    }

    public void printHeader() {
        System.out.println("Iteration #\tObjective Value\tPoint");
        System.out.println("===========\t===============\t=====");
    }

    public void printIteration() {
        System.out.format("%d\t\t\t%.10f\t%.5f\n", currentIteration, currentObjectiveValue, currentPoint.getEntry(0));
    }

    public RealVector solve() {
        printHeader();
        do {
            updateDirection();
            updateStepSize();

            previousDirection = currentDirection;
            previousStepSize = currentStepSize;
            previousPoint = currentPoint;
            previousObjectiveValue = currentObjectiveValue;

            updatePoint();

            currentObjectiveValue = objective.computeValue(currentPoint);
            currentIteration++;

            printIteration();
        } while (!checkForConvergence());

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

        return currentPoint;
    }

    public void setLineSearch(LineSearch lineSearch) {
        this.lineSearch = lineSearch;
    }
}
