package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.optimization.AdaptiveGradientSolver;
import org.platanios.learn.optimization.StochasticSolverStepSize;
import org.platanios.learn.serialization.UnsafeSerializationUtilities;

import java.io.IOException;
import java.io.InputStream;
import java.io.InvalidObjectException;
import java.io.OutputStream;
import java.util.Random;
import java.util.function.Function;

/**
 * This class implements a binary logistic regression model that is trained using the adaptive gradient algorithm.
 *
 * @author Emmanouil Antonios Platanios
 */
public class LogisticRegressionAdaGrad extends AbstractTrainableLogisticRegression {
    private boolean sampleWithReplacement;
    private int maximumNumberOfIterations;
    private int maximumNumberOfIterationsWithNoPointChange;
    private double pointChangeTolerance;
    private boolean checkForPointConvergence;
    private Function<Vector, Boolean> additionalCustomConvergenceCriterion;
    private int batchSize;
    private StochasticSolverStepSize stepSize;
    private double[] stepSizeParameters;
    private Random random;

    /**
     * This abstract class needs to be extended by the builder of its parent binary logistic regression class. It
     * provides an implementation for those parts of those builders that are common. This is basically part of a small
     * "hack" so that we can have inheritable builder classes.
     *
     * @param   <T> This type corresponds to the type of the final object to be built. That is, the super class of the
     *              builder class that extends this class, which in this case will be the
     *              {@link LogisticRegressionAdaGrad} class.
     */
    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractTrainableLogisticRegression.AbstractBuilder<T> {
        protected boolean sampleWithReplacement = false;
        protected int maximumNumberOfIterations = 10000;
        protected int maximumNumberOfIterationsWithNoPointChange = 5;
        protected double pointChangeTolerance = 1e-10;
        protected boolean checkForPointConvergence = true;
        private Function<Vector, Boolean> additionalCustomConvergenceCriterion = currentPoint -> false;
        protected int batchSize = 100;
        protected StochasticSolverStepSize stepSize = StochasticSolverStepSize.SCALED;
        protected double[] stepSizeParameters = new double[] { 10, 0.75 };
        protected Random random = new Random();

        protected AbstractBuilder(int numberOfFeatures) {
            super(numberOfFeatures);
        }

        protected AbstractBuilder(int numberOfFeatures, Vector weights) {
            super(numberOfFeatures, weights);
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

        public T additionalCustomConvergenceCriterion(Function<Vector, Boolean> additionalCustomConvergenceCriterion) {
            this.additionalCustomConvergenceCriterion = additionalCustomConvergenceCriterion;
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
        
        public T random(Random random) {
        	this.random = random;
        	return self();
        }

        @Override
        public T setParameter(String name, Object value) {
            switch (name) {
                case "sampleWithReplacement":
                    sampleWithReplacement = (boolean) value;
                    break;
                case "maximumNumberOfIterations":
                    maximumNumberOfIterations = (int) value;
                    break;
                case "maximumNumberOfIterationsWithNoPointChange":
                    maximumNumberOfIterationsWithNoPointChange = (int) value;
                    break;
                case "pointChangeTolerance":
                    pointChangeTolerance = (double) value;
                    break;
                case "checkForPointConvergence":
                    checkForPointConvergence = (boolean) value;
                    break;
                case "batchSize":
                    batchSize = (int) value;
                    break;
                case "stepSize":
                    stepSize = (StochasticSolverStepSize) value;
                    break;
                case "stepSizeParameters":
                    stepSizeParameters = (double[]) value;
                    break;
                default:
                    super.setParameter(name, value);
            }
            return self();
        }

        @Override
        public LogisticRegressionAdaGrad build() {
            return new LogisticRegressionAdaGrad(this);
        }
    }

    /**
     * The builder class for this class. This is basically part of a small "hack" so that we can have inheritable
     * builder classes.
     */
    public static class Builder extends AbstractBuilder<Builder>
            implements TrainableClassifier.Builder<Vector, Double> {
        /**
         * Constructs a builder object for a binary logistic regression model that will be trained with the provided
         * training data using the stochastic gradient descent algorithm.
         *
         * @param   numberOfFeatures    The number of features used.
         */
        public Builder(int numberOfFeatures) {
            super(numberOfFeatures);
        }

        public Builder(int numberOfFeatures, Vector weights) {
            super(numberOfFeatures, weights);
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
    private LogisticRegressionAdaGrad(AbstractBuilder<?> builder) {
        super(builder);

        sampleWithReplacement = builder.sampleWithReplacement;
        maximumNumberOfIterations = builder.maximumNumberOfIterations;
        maximumNumberOfIterationsWithNoPointChange = builder.maximumNumberOfIterationsWithNoPointChange;
        pointChangeTolerance = builder.pointChangeTolerance;
        checkForPointConvergence = builder.checkForPointConvergence;
        additionalCustomConvergenceCriterion = builder.additionalCustomConvergenceCriterion;
        batchSize = builder.batchSize;
        stepSize = builder.stepSize;
        stepSizeParameters = builder.stepSizeParameters;
        random = builder.random;
    }

    @Override
    public ClassifierType type() {
        return ClassifierType.LOGISTIC_REGRESSION_ADAGRAD;
    }

    /** {@inheritDoc} */
    @Override
    protected void train() {
        weights =  new AdaptiveGradientSolver.Builder(new StochasticLikelihoodFunction(random), weights)
                .sampleWithReplacement(sampleWithReplacement)
                .maximumNumberOfIterations(maximumNumberOfIterations)
                .maximumNumberOfIterationsWithNoPointChange(maximumNumberOfIterationsWithNoPointChange)
                .pointChangeTolerance(pointChangeTolerance)
                .checkForPointConvergence(checkForPointConvergence)
                .additionalCustomConvergenceCriterion(additionalCustomConvergenceCriterion)
                .batchSize(batchSize)
                .stepSize(stepSize)
                .stepSizeParameters(stepSizeParameters)
                .useL1Regularization(useL1Regularization)
                .l1RegularizationWeight(l1RegularizationWeight)
                .useL2Regularization(useL2Regularization)
                .l2RegularizationWeight(l2RegularizationWeight)
                .loggingLevel(loggingLevel)
                .build()
                .solve();
    }

    /** {@inheritDoc} */
    @Override
    public void write(OutputStream outputStream, boolean includeType) throws IOException {
        super.write(outputStream, includeType);

        UnsafeSerializationUtilities.writeBoolean(outputStream, sampleWithReplacement);
        UnsafeSerializationUtilities.writeInt(outputStream, maximumNumberOfIterations);
        UnsafeSerializationUtilities.writeInt(outputStream, maximumNumberOfIterationsWithNoPointChange);
        UnsafeSerializationUtilities.writeDouble(outputStream, pointChangeTolerance);
        UnsafeSerializationUtilities.writeBoolean(outputStream, checkForPointConvergence);
        UnsafeSerializationUtilities.writeInt(outputStream, batchSize);
        UnsafeSerializationUtilities.writeInt(outputStream, stepSize.ordinal());
        UnsafeSerializationUtilities.writeInt(outputStream, stepSizeParameters.length);
        UnsafeSerializationUtilities.writeDoubleArray(outputStream, stepSizeParameters);
    }

    public static LogisticRegressionAdaGrad read(InputStream inputStream, boolean includeType) throws IOException {
        if (includeType) {
            ClassifierType classifierType = ClassifierType.values()[UnsafeSerializationUtilities.readInt(inputStream)];
            if (!ClassifierType.LOGISTIC_REGRESSION_ADAGRAD
                    .getStorageCompatibleTypes()
                    .contains(classifierType))
                throw new InvalidObjectException("The stored classifier is of type " + classifierType.name() + "!");
        }
        int numberOfFeatures = UnsafeSerializationUtilities.readInt(inputStream);
        boolean sparse = UnsafeSerializationUtilities.readBoolean(inputStream);
        Vector weights = Vectors.build(inputStream);
        boolean usel1Regularization = UnsafeSerializationUtilities.readBoolean(inputStream);
        double l1RegularizationWeight = UnsafeSerializationUtilities.readDouble(inputStream);
        boolean usel2Regularization = UnsafeSerializationUtilities.readBoolean(inputStream);
        double l2RegularizationWeight = UnsafeSerializationUtilities.readDouble(inputStream);
        int loggingLevel = UnsafeSerializationUtilities.readInt(inputStream);
        boolean sampleWithReplacement = UnsafeSerializationUtilities.readBoolean(inputStream);
        int maximumNumberOfIterations = UnsafeSerializationUtilities.readInt(inputStream);
        int maximumNumberOfIterationsWithNoPointChange = UnsafeSerializationUtilities.readInt(inputStream);
        double pointChangeTolerance = UnsafeSerializationUtilities.readDouble(inputStream);
        boolean checkForPointConvergence = UnsafeSerializationUtilities.readBoolean(inputStream);
        int batchSize = UnsafeSerializationUtilities.readInt(inputStream);
        StochasticSolverStepSize stepSize =
                StochasticSolverStepSize.values()[UnsafeSerializationUtilities.readInt(inputStream)];
        int numberOfStepSizeParameters = UnsafeSerializationUtilities.readInt(inputStream);
        double[] stepSizeParameters =
                UnsafeSerializationUtilities.readDoubleArray(inputStream, numberOfStepSizeParameters, 4);
        return new Builder(numberOfFeatures, weights)
                .sparse(sparse)
                .useL1Regularization(usel1Regularization)
                .l1RegularizationWeight(l1RegularizationWeight)
                .useL2Regularization(usel2Regularization)
                .l2RegularizationWeight(l2RegularizationWeight)
                .loggingLevel(loggingLevel)
                .sampleWithReplacement(sampleWithReplacement)
                .maximumNumberOfIterations(maximumNumberOfIterations)
                .maximumNumberOfIterationsWithNoPointChange(maximumNumberOfIterationsWithNoPointChange)
                .pointChangeTolerance(pointChangeTolerance)
                .checkForPointConvergence(checkForPointConvergence)
                .batchSize(batchSize)
                .stepSize(stepSize)
                .stepSizeParameters(stepSizeParameters)
                .build();
    }
}
