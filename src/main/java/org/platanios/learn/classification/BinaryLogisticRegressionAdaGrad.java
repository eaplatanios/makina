package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.AdaptiveGradientSolver;
import org.platanios.learn.optimization.StochasticSolverStepSize;

/**
 * This class implements a binary logistic regression model that is trained using the adaptive gradient algorithm.
 *
 * @author Emmanouil Antonios Platanios
 */
public class BinaryLogisticRegressionAdaGrad extends AbstractBinaryLogisticRegression {
    /** The adaptive gradient solver that is used to train this binary logistic regression model. */
    private final AdaptiveGradientSolver solver;

    /**
     * This abstract class needs to be extended by the builder of its parent binary logistic regression class. It
     * provides an implementation for those parts of those builders that are common. This is basically part of a small
     * "hack" so that we can have inheritable builder classes.
     *
     * @param   <T> This type corresponds to the type of the final object to be built. That is, the super class of the
     *              builder class that extends this class, which in this case will be the
     *              {@link org.platanios.learn.classification.BinaryLogisticRegressionAdaGrad} class.
     */
    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractBinaryLogisticRegression.AbstractBuilder<T> {
        protected boolean sampleWithReplacement = false;
        protected int maximumNumberOfIterations = 10000;
        protected int maximumNumberOfIterationsWithNoPointChange = 5;
        protected double pointChangeTolerance = 1e-10;
        protected boolean checkForPointConvergence = true;
        protected int batchSize = 1000;
        protected StochasticSolverStepSize stepSize = StochasticSolverStepSize.SCALED;
        protected double[] stepSizeParameters = new double[] { 10, 0.75 };

        public AbstractBuilder(DataInstance<Vector, Integer>[] trainingData) {
            super(trainingData);
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

        public BinaryLogisticRegressionAdaGrad build() {
            return new BinaryLogisticRegressionAdaGrad(this);
        }
    }

    /**
     * The builder class for this class. This is basically part of a small "hack" so that we can have inheritable
     * builder classes.
     */
    public static class Builder extends AbstractBuilder<Builder> {
        /**
         * Constructs a builder object for a binary logistic regression model that will be trained with the provided
         * training data using the adaptive gradient algorithm.
         *
         * @param   trainingData    The training data with which the binary logistic regression model to be built by
         *                          this builder will be trained.
         */
        public Builder(DataInstance<Vector, Integer>[] trainingData) {
            super(trainingData);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }
    }

    /**
     * Constructs a binary logistic regression object that uses the adaptive gradient algorithm to train the model,
     * given an appropriate builder object. This constructor can only be used from within the builder class of this
     * class.
     *
     * @param   builder The builder object to use.
     */
    private BinaryLogisticRegressionAdaGrad(AbstractBuilder<?> builder) {
        super(builder);

        solver = new AdaptiveGradientSolver.Builder(new StochasticLikelihoodFunction(), weights)
                .sampleWithReplacement(builder.sampleWithReplacement)
                .maximumNumberOfIterations(builder.maximumNumberOfIterations)
                .maximumNumberOfIterationsWithNoPointChange(builder.maximumNumberOfIterationsWithNoPointChange)
                .pointChangeTolerance(builder.pointChangeTolerance)
                .checkForPointConvergence(builder.checkForPointConvergence)
                .batchSize(builder.batchSize)
                .stepSize(builder.stepSize)
                .stepSizeParameters(builder.stepSizeParameters)
                .useL1Regularization(builder.useL1Regularization)
                .l1RegularizationWeight(builder.l1RegularizationWeight)
                .useL2Regularization(builder.useL2Regularization)
                .l2RegularizationWeight(builder.l2RegularizationWeight)
                .build();
    }

    /** {@inheritDoc} */
    @Override
    public void train() {
        weights = solver.solve();
    }
}
