package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.DenseVector;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorNorm;
import org.platanios.learn.optimization.function.AbstractFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
abstract class AbstractIterativeSolver implements Solver {
    private final int maximumNumberOfIterations;
    private final int maximumNumberOfFunctionEvaluations;
    private final double pointChangeTolerance;
    private final double objectiveChangeTolerance;
    private final double gradientTolerance;
    private final boolean checkForPointConvergence;
    private final boolean checkForObjectiveConvergence;
    private final boolean checkForGradientConvergence;

    private double pointChange;
    private double objectiveChange;
    private double gradientNorm;

    private boolean pointConverged = false;
    private boolean objectiveConverged = false;
    private boolean gradientConverged = false;

    final AbstractFunction objective;

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

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>> {
        protected abstract T self();

        protected final AbstractFunction objective;
        protected final double[] initialPoint;

        protected int maximumNumberOfIterations = 10000;
        protected int maximumNumberOfFunctionEvaluations = 1000000;
        protected double pointChangeTolerance = 1e-10;
        protected double objectiveChangeTolerance = 1e-10;
        protected double gradientTolerance = 1e-6;
        protected boolean checkForPointConvergence = true;
        protected boolean checkForObjectiveConvergence = true;
        protected boolean checkForGradientConvergence = true;

        protected AbstractBuilder(AbstractFunction objective,
                                  double[] initialPoint) {
            this.objective = objective;
            this.initialPoint = initialPoint;
        }

        public T maximumNumberOfIterations(int maximumNumberOfIterations) {
            this.maximumNumberOfIterations = maximumNumberOfIterations;
            return self();
        }

        public T maximumNumberOfFunctionEvaluations(int maximumNumberOfFunctionEvaluations) {
            this.maximumNumberOfFunctionEvaluations = maximumNumberOfFunctionEvaluations;
            return self();
        }

        public T pointChangeTolerance(double pointChangeTolerance) {
            this.pointChangeTolerance = pointChangeTolerance;
            return self();
        }

        public T objectiveChangeTolerance(double objectiveChangeTolerance) {
            this.objectiveChangeTolerance = objectiveChangeTolerance;
            return self();
        }

        public T gradientTolerance(double gradientTolerance) {
            this.gradientTolerance = gradientTolerance;
            return self();
        }

        public T checkForPointConvergence(boolean checkForPointConvergence) {
            this.checkForPointConvergence = checkForPointConvergence;
            return self();
        }

        public T checkForObjectiveConvergence(boolean checkForObjectiveConvergence) {
            this.checkForObjectiveConvergence = checkForObjectiveConvergence;
            return self();
        }

        public T checkForGradientConvergence(boolean checkForGradientConvergence) {
            this.checkForGradientConvergence = checkForGradientConvergence;
            return self();
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(AbstractFunction objective,
                       double[] initialPoint) {
            super(objective, initialPoint);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    protected AbstractIterativeSolver(AbstractBuilder<?> builder) {
        objective = builder.objective;
        maximumNumberOfIterations = builder.maximumNumberOfIterations;
        maximumNumberOfFunctionEvaluations = builder.maximumNumberOfFunctionEvaluations;
        pointChangeTolerance = builder.pointChangeTolerance;
        objectiveChangeTolerance = builder.objectiveChangeTolerance;
        gradientTolerance = builder.gradientTolerance;
        checkForPointConvergence = builder.checkForPointConvergence;
        checkForObjectiveConvergence = builder.checkForObjectiveConvergence;
        checkForGradientConvergence = builder.checkForGradientConvergence;

        currentPoint = new DenseVector(builder.initialPoint);
        currentGradient = objective.getGradient(currentPoint);
        currentObjectiveValue = objective.getValue(currentPoint);
        currentIteration = 0;
    }

    @Override
    public Vector solve() {
        printHeader();
        while (!checkTerminationConditions()) {
            performIterationUpdates();
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
                pointChange = currentPoint.subtract(previousPoint).norm(VectorNorm.L2);
                pointConverged = pointChange <= pointChangeTolerance;
            }

            if (checkForObjectiveConvergence) {
                objectiveChange = Math.abs((previousObjectiveValue - currentObjectiveValue) / previousObjectiveValue);
                objectiveConverged = objectiveChange <= objectiveChangeTolerance;
            }

            if (checkForGradientConvergence) {
                if (this instanceof NonlinearConjugateGradientSolver) {
                    gradientNorm =
                            Math.abs(currentGradient.max()) / (1 + Math.abs(currentObjectiveValue));
                    gradientConverged = gradientNorm <= gradientTolerance;
                } else {
                    gradientNorm = currentGradient.norm(VectorNorm.L2);
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

    public abstract void performIterationUpdates();
}
