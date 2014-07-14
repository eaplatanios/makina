package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.function.AbstractFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
abstract class AbstractIterativeSolver implements Solver {
    private int maximumNumberOfIterations = 10000;
    private int maximumNumberOfFunctionEvaluations = 1000000;

    private double pointChangeTolerance = 1e-10;
    private double objectiveChangeTolerance = 1e-10;
    private double gradientTolerance = 1e-5;

    private boolean checkForPointConvergence = true;
    private boolean checkForObjectiveConvergence = true;
    private boolean checkForGradientConvergence = true;

    private double pointChange;
    private double objectiveChange;
    private double gradientNorm;

    private boolean pointConverged = false;
    private boolean objectiveConverged = false;
    private boolean gradientConverged = false;

    AbstractFunction objective;
    int currentIteration;
    Vector currentPoint;
    Vector previousPoint;
    Vector currentGradient;
    Vector previousGradient;
    Vector currentDirection;
    Vector previousDirection;
    double currentStepSize;
    double previousStepSize;
    double currentObjectiveValue;
    double previousObjectiveValue;

    public static abstract class Builder {
        protected final AbstractFunction objective;
        protected final double[] initialPoint;

        protected Builder(AbstractFunction objective,
                          double[] initialPoint) {
            this.objective = objective;
            this.initialPoint = initialPoint;
        }
    }

    AbstractIterativeSolver(Builder builder) {
        this.objective = builder.objective;
        currentPoint = new Vector(builder.initialPoint);
        currentGradient = objective.getGradient(currentPoint);
        currentObjectiveValue = objective.getValue(currentPoint);
        currentIteration = 0;
    }

    @Override
    public Vector solve() {
        printHeader();
        while (!checkTerminationConditions()) {
            iterationUpdate();
            currentIteration++;
            printIteration();
        }
        printTerminationMessage();
        return currentPoint;
    }

    public boolean checkTerminationConditions() {
        if (currentIteration > 0) {
            if (currentIteration >= maximumNumberOfIterations) {
                return true;
            }

            if (objective.getNumberOfFunctionEvaluations() >= maximumNumberOfFunctionEvaluations) {
                return true;
            }

            if (checkForPointConvergence) {
                pointChange = currentPoint.subtract(previousPoint).computeL2Norm();
                pointConverged = pointChange <= pointChangeTolerance;
            }

            if (checkForObjectiveConvergence) {
                objectiveChange = Math.abs((previousObjectiveValue - currentObjectiveValue) / previousObjectiveValue);
                objectiveConverged = objectiveChange <= objectiveChangeTolerance;
            }

            if (checkForGradientConvergence) {
                if (this instanceof NonlinearConjugateGradientSolver) {
                    gradientNorm =
                            Math.abs(currentGradient.getMaximumValue()) / (1 + Math.abs(currentObjectiveValue));
                    gradientConverged = gradientNorm <= gradientTolerance;
                } else {
                    gradientNorm = currentGradient.computeL2Norm();
                    gradientConverged = gradientNorm <= gradientTolerance;
                }
            }

            return (checkForPointConvergence && pointConverged)
                    || (checkForObjectiveConvergence && objectiveConverged)
                    || (checkForGradientConvergence && gradientConverged);
        } else {
            return false;
        }
    }

    public void printHeader() {
        System.out.println("|----------------" +
                                   "----------------" +
                                   "-----------------------" +
                                   "-----------------------" +
                                   "-----------------------" +
                                   "----------------------|");
        System.out.format("| %13s | %13s | %20s | %20s | %20s | %20s |%n",
                          "Iteration #",
                          "Func. Eval. #",
                          "Objective Value",
                          "Objective Change",
                          "Point Change",
                          "Gradient Norm");
        System.out.println("|===============|" +
                                   "===============|" +
                                   "======================|" +
                                   "======================|" +
                                   "======================|" +
                                   "======================|");
    }

    public void printIteration() {
        System.out.format("| %13d | %13s | %20s | %20s | %20s | %20s |%n",
                          currentIteration,
                          objective.getNumberOfFunctionEvaluations(),
                          DECIMAL_FORMAT.format(currentObjectiveValue),
                          DECIMAL_FORMAT.format(objectiveChange),
                          DECIMAL_FORMAT.format(pointChange),
                          DECIMAL_FORMAT.format(gradientNorm));
    }

    public void printTerminationMessage() {
        System.out.println("|----------------" +
                                   "----------------" +
                                   "-----------------------" +
                                   "-----------------------" +
                                   "-----------------------" +
                                   "----------------------|\n");

        if (pointConverged) {
            System.out.println("The L2 norm of the point change, "
                                       + DECIMAL_FORMAT.format(pointChange)
                                       + ", was below the convergence threshold of "
                                       + DECIMAL_FORMAT.format(pointChangeTolerance)
                                       + "!");
        }
        if (objectiveConverged) {
            System.out.println("The relative change of the objective function value, "
                                       + DECIMAL_FORMAT.format(objectiveChange)
                                       + ", was below the convergence threshold of "
                                       + DECIMAL_FORMAT.format(objectiveChangeTolerance)
                                       + "!");
        }
        if (gradientConverged) {
            System.out.println("The gradient norm became "
                                       + DECIMAL_FORMAT.format(gradientNorm)
                                       + ", which is less than the convergence threshold of "
                                       + DECIMAL_FORMAT.format(gradientTolerance)
                                       + "!");
        }
        if (currentIteration >= maximumNumberOfIterations) {
            System.out.println("Reached the maximum number of allowed iterations ("
                                       + maximumNumberOfIterations
                                       + ")!");
        }

        if (objective.getNumberOfFunctionEvaluations() >= maximumNumberOfFunctionEvaluations) {
            System.out.println("Reached the maximum number of allowed objective function evaluations ("
                                       + maximumNumberOfFunctionEvaluations
                                       + ")!");
        }
    }

    public abstract void iterationUpdate();

    public int getMaximumNumberOfIterations() {
        return maximumNumberOfIterations;
    }

    public void setMaximumNumberOfIterations(int maximumNumberOfIterations) {
        this.maximumNumberOfIterations = maximumNumberOfIterations;
    }

    public int getMaximumNumberOfFunctionEvaluations() {
        return maximumNumberOfFunctionEvaluations;
    }

    public void setMaximumNumberOfFunctionEvaluations(int maximumNumberOfFunctionEvaluations) {
        this.maximumNumberOfFunctionEvaluations = maximumNumberOfFunctionEvaluations;
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

    public boolean isCheckForPointConvergence() {
        return checkForPointConvergence;
    }

    public void setCheckForPointConvergence(boolean checkForPointConvergence) {
        this.checkForPointConvergence = checkForPointConvergence;
    }

    public boolean isCheckForObjectiveConvergence() {
        return checkForObjectiveConvergence;
    }

    public void setCheckForObjectiveConvergence(boolean checkForObjectiveConvergence) {
        this.checkForObjectiveConvergence = checkForObjectiveConvergence;
    }

    public boolean isCheckForGradientConvergence() {
        return checkForGradientConvergence;
    }

    public void setCheckForGradientConvergence(boolean checkForGradientConvergence) {
        this.checkForGradientConvergence = checkForGradientConvergence;
    }
}
