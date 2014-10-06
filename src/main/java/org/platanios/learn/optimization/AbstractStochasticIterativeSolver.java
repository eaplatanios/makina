package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorNorm;
import org.platanios.learn.optimization.function.AbstractStochasticFunction;

/**
 * TODO: Generalize the support for regularization. I could add a addRegularization(RegularizationType) method to add
 * many and any kind of regularizations.
 *
 * @author Emmanouil Antonios Platanios
 */
abstract class AbstractStochasticIterativeSolver implements Solver {
    private final int maximumNumberOfIterations;
    private final int maximumNumberOfIterationsWithNoPointChange;
    private final double pointChangeTolerance;
    private final boolean checkForPointConvergence;
    private final int batchSize;
    private final StochasticSolverStepSize stepSize;
    private final double[] stepSizeParameters;
    /** Indicates whether /(L_1/) regularization is used. */
    protected final boolean useL1Regularization;
    /** The /(L_1/) regularization weight used. This variable is only used when {@link #useL1Regularization} is set to
     * true. */
    protected final double l1RegularizationWeight;
    /** Indicates whether /(L_2/) regularization is used. */
    private final boolean useL2Regularization;
    /** The /(L_2/) regularization weight used. This variable is only used when {@link #useL2Regularization} is set
     * to true. */
    private final double l2RegularizationWeight;

    private double pointChange;
    private int numberOfIterationsWithNoPointChange = 0;
    private boolean pointConverged = false;

    final AbstractStochasticFunction objective;

    int currentIteration;
    Vector currentPoint;
    Vector previousPoint;
    Vector currentGradient;
    Vector currentDirection;
    double currentStepSize;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>> {
        protected abstract T self();

        protected final AbstractStochasticFunction objective;
        protected final Vector initialPoint;

        protected boolean sampleWithReplacement = true;
        protected int maximumNumberOfIterations = 10000;
        protected int maximumNumberOfIterationsWithNoPointChange = 1;
        protected double pointChangeTolerance = 1e-10;
        protected boolean checkForPointConvergence = true;
        protected int batchSize = 100;
        protected StochasticSolverStepSize stepSize = StochasticSolverStepSize.SCALED;
        protected double[] stepSizeParameters = new double[] { 10, 0.75 };
        /** Indicates whether /(L_1/) regularization is used. */
        private boolean useL1Regularization = false;
        /** The /(L_1/) regularization weight used. This variable is only used when {@link #useL1Regularization} is set
         * to true. */
        private double l1RegularizationWeight = 1;
        /** Indicates whether /(L_2/) regularization is used. */
        private boolean useL2Regularization = false;
        /** The /(L_2/) regularization weight used. This variable is only used when {@link #useL2Regularization} is set
         * to true. */
        private double l2RegularizationWeight = 1;

        protected AbstractBuilder(AbstractStochasticFunction objective,
                                  Vector initialPoint) {
            this.objective = objective;
            this.initialPoint = initialPoint;
        }

        public T sampleWithReplacement(boolean sampleWithReplacement) {
            this.sampleWithReplacement = sampleWithReplacement;
            return self();
        }

        public T maximumNumberOfIterations(int maximumNumberOfIterations) {
            this.maximumNumberOfIterations = maximumNumberOfIterations;
            return self();
        }

        public T maximumNumberOfIterationsWithNoPointChange(int maximumNumberOfIterationsWithNoPointChange) {
            this.maximumNumberOfIterationsWithNoPointChange = maximumNumberOfIterationsWithNoPointChange;
            return self();
        }

        public T pointChangeTolerance(double pointChangeTolerance) {
            this.pointChangeTolerance = pointChangeTolerance;
            return self();
        }

        public T checkForPointConvergence(boolean checkForPointConvergence) {
            this.checkForPointConvergence = checkForPointConvergence;
            return self();
        }

        public T batchSize(int batchSize) {
            this.batchSize = batchSize;
            return self();
        }

        public T stepSize(StochasticSolverStepSize stepSize) {
            this.stepSize = stepSize;
            return self();
        }

        public T stepSizeParameters(double... stepSizeParameters) {
            this.stepSizeParameters = stepSizeParameters;
            return self();
        }

        /**
         * Sets the {@link #useL1Regularization} field that indicates whether /(L_1/) regularization is used.
         *
         * @param   useL1Regularization The value to which to set the {@link #useL1Regularization} field.
         * @return                      This builder object itself. That is done so that we can use a nice and
         *                              expressive code format when we build objects using this builder class.
         */
        public T useL1Regularization(boolean useL1Regularization) {
            this.useL1Regularization = useL1Regularization;
            return self();
        }

        /**
         * Sets the {@link #l1RegularizationWeight} field that contains the value of the /(L_1/) regularization weight
         * used. This variable is only used when {@link #useL1Regularization} is set to true.
         *
         * @param   l1RegularizationWeight  The value to which to set the {@link #l1RegularizationWeight} field.
         * @return                          This builder object itself. That is done so that we can use a nice and
         *                                  expressive code format when we build objects using this builder class.
         */
        public T l1RegularizationWeight(double l1RegularizationWeight) {
            this.l1RegularizationWeight = l1RegularizationWeight;
            return self();
        }

        /**
         * Sets the {@link #useL2Regularization} field that indicates whether /(L_2/) regularization is used.
         *
         * @param   usel2Regularization The value to which to set the {@link #useL2Regularization} field.
         * @return                      This builder object itself. That is done so that we can use a nice and
         *                              expressive code format when we build objects using this builder class.
         */
        public T useL2Regularization(boolean usel2Regularization) {
            this.useL2Regularization = usel2Regularization;
            return self();
        }

        /**
         * Sets the {@link #l2RegularizationWeight} field that contains the value of the /(L_2/) regularization weight
         * used. This variable is only used when {@link #useL2Regularization} is set to true.
         *
         * @param   l2RegularizationWeight  The value to which to set the {@link #l2RegularizationWeight} field.
         * @return                          This builder object itself. That is done so that we can use a nice and
         *                                  expressive code format when we build objects using this builder class.
         */
        public T l2RegularizationWeight(double l2RegularizationWeight) {
            this.l2RegularizationWeight = l2RegularizationWeight;
            return self();
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(AbstractStochasticFunction objective,
                       Vector initialPoint) {
            super(objective, initialPoint);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    protected AbstractStochasticIterativeSolver(AbstractBuilder<?> builder) {
        objective = builder.objective;
        objective.setSampleWithReplacement(builder.sampleWithReplacement);
        maximumNumberOfIterations = builder.maximumNumberOfIterations;
        maximumNumberOfIterationsWithNoPointChange = builder.maximumNumberOfIterationsWithNoPointChange;
        pointChangeTolerance = builder.pointChangeTolerance;
        checkForPointConvergence = builder.checkForPointConvergence;
        batchSize = builder.batchSize;
        stepSize = builder.stepSize;
        stepSizeParameters = builder.stepSizeParameters;
        useL1Regularization = builder.useL1Regularization;
        l1RegularizationWeight = builder.l1RegularizationWeight;
        useL2Regularization = builder.useL2Regularization;
        l2RegularizationWeight = builder.l2RegularizationWeight;
        currentPoint = builder.initialPoint;
        currentGradient = objective.getGradientEstimate(currentPoint, batchSize);
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

    public void updateStepSize() {
        currentStepSize = stepSize.compute(currentIteration, stepSizeParameters);
    }

    public void performIterationUpdates() {
        updateDirection();
        updateStepSize();
        previousPoint = currentPoint;
        updatePoint();
        currentGradient = objective.getGradientEstimate(currentPoint, batchSize);
        if (useL2Regularization) {
            currentGradient.addInPlace(currentPoint.mult(2 * l2RegularizationWeight));
        }
    }

    public boolean checkTerminationConditions() {
        if (currentIteration > 0) {
            if (currentIteration >= maximumNumberOfIterations) {
                return true;
            }

            if (checkForPointConvergence) {
                pointChange = currentPoint.sub(previousPoint).norm(VectorNorm.L2);
                numberOfIterationsWithNoPointChange =
                        (pointChange <= pointChangeTolerance) ? numberOfIterationsWithNoPointChange + 1 : 0;
                if (numberOfIterationsWithNoPointChange >= maximumNumberOfIterationsWithNoPointChange) {
                    pointConverged = true;
                }
            }

            return checkForPointConvergence && pointConverged;
        } else {
            return false;
        }
    }

    public void printHeader() {
        System.out.println("|----------------" +
                                   "----------------------|");
        System.out.format("| %13s | %20s |%n",
                          "Iteration #",
                          "Point Change");
        System.out.println("|===============|" +
                                   "======================|");
    }

    public void printIteration() {
        System.out.format("| %13d | %20s |%n",
                          currentIteration,
                          DECIMAL_FORMAT.format(pointChange));
    }

    public void printTerminationMessage() {
        System.out.println("|----------------" +
                                   "----------------------|\n");

        if (pointConverged) {
            System.out.println("The L2 norm of the point change, "
                                       + DECIMAL_FORMAT.format(pointChange)
                                       + ", was below the convergence threshold of "
                                       + DECIMAL_FORMAT.format(pointChangeTolerance)
                                       + " for more than "
                                       + maximumNumberOfIterationsWithNoPointChange
                                       + " iterations!");
        }
        if (currentIteration >= maximumNumberOfIterations) {
            System.out.println("Reached the maximum number of allowed iterations ("
                                       + maximumNumberOfIterations
                                       + ")!");
        }
    }

    /**
     *
     *
     * Note: Care must be taken when implementing this method to include the relevant cases for when \(L_1\) or \(L_2\)
     * regularization is used.
     */
    public abstract void updateDirection();

    /**
     *
     *
     * Note 1: Care must be taken when implementing this method to include the relevant cases for when \(L_1\) or \(L_2\)
     * regularization is used.
     *
     * Note 2: Care must be taken when implementing this method because the previousPoint variable is simply updated to
     * point to currentPoint, at the beginning of each iteration. That means that when the new value is computed, a new
     * object has to be instantiated for holding that values.
     */
    public abstract void updatePoint();
}
