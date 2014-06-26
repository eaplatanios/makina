package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;
import org.platanios.learn.optimization.function.QuadraticFunction;
import org.platanios.learn.optimization.linesearch.ExactLineSearch;
import org.platanios.learn.optimization.linesearch.LineSearch;
import org.platanios.learn.optimization.linesearch.StepSizeInitializationMethod;
import org.platanios.learn.optimization.linesearch.StrongWolfeLineSearch;

/**
 * @author Emmanouil Antonios Platanios
 */
public class FletcherReevesSolver extends AbstractSolver {
    private final Function objective;

    /** Default value: If quadratic or linear function it is ExactLineSearch, otherwise it is StrongWolfeLineSearch
     * with CONSERVE_FIRST_ORDER_CHANGE for the step size initialization method. */
    private LineSearch lineSearch;

    private RealVector previousPoint;
    private RealVector currentGradient;
    private RealVector previousGradient;
    private RealVector currentDirection;
    private RealVector previousDirection;
    private double currentStepSize;
    private double previousStepSize;
    private double currentObjectiveValue;
    private double previousObjectiveValue;

    private double pointChangeTolerance = 1e-10;
    private double objectiveChangeTolerance = 1e-10;
    private double gradientTolerance = 1e-10;

    private double pointChange;
    private double objectiveChange;
    private double gradientNorm;

    private boolean pointConverged = false;
    private boolean gradientConverged = false;
    private boolean objectiveConverged = false;

    // The following variables are used locally within iteration but are initialized here in order to make the code more
    // clear.
    double gradientNormsRatio;

    public FletcherReevesSolver(Function objective,
                                double[] initialPoint) {
        this.objective = objective;
        currentPoint = new ArrayRealVector(initialPoint);
        currentGradient = objective.computeGradient(currentPoint);
        currentDirection = currentGradient.mapMultiply(-1);
        currentObjectiveValue = objective.computeValue(currentPoint);
        currentIteration = 0;

        if (objective instanceof QuadraticFunction) {
            this.lineSearch = new ExactLineSearch((QuadraticFunction) objective);
        } else {
            this.lineSearch = new StrongWolfeLineSearch(
                    objective,
                    StepSizeInitializationMethod.CONSERVE_FIRST_ORDER_CHANGE,
                    1e-4,
                    0.5,
                    10
            );
        }
    }

    public boolean checkTerminationConditions() {
        if (currentIteration > 0) {
            pointChange = currentPoint.subtract(previousPoint).getNorm();
            objectiveChange = Math.abs((previousObjectiveValue - currentObjectiveValue) / previousObjectiveValue);
            gradientNorm = currentGradient.getNorm();

            pointConverged = pointChange <= pointChangeTolerance;
            objectiveConverged = objectiveChange <= objectiveChangeTolerance;
            gradientConverged = gradientNorm <= gradientTolerance;

            return pointConverged || objectiveConverged || gradientConverged;
        } else {
            return false;
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
        gradientNormsRatio =
                currentGradient.dotProduct(currentGradient) / previousGradient.dotProduct(previousGradient);
        currentDirection = currentGradient.mapMultiply(-1).add(previousDirection.mapMultiply(gradientNormsRatio));
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
