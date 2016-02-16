//package org.platanios.learn.classification;
//
//import org.platanios.learn.math.matrix.Vector;
//import org.platanios.learn.math.matrix.Vectors;
//import org.platanios.learn.optimization.QuasiNewtonSolver;
//import org.platanios.learn.optimization.linesearch.LineSearch;
//import org.platanios.learn.serialization.UnsafeSerializationUtilities;
//
//import java.io.IOException;
//import java.io.InputStream;
//import java.io.InvalidObjectException;
//import java.io.OutputStream;
//import java.util.function.Function;
//
///**
// * This class implements a binary logistic regression model that is trained using the adaptive gradient algorithm.
// *
// * @author Emmanouil Antonios Platanios
// */
//public class LogisticRegressionQuasiNewton extends AbstractTrainableLogisticRegression {
//    private final int maximumNumberOfIterations;
//    private final int maximumNumberOfFunctionEvaluations;
//    private final double pointChangeTolerance;
//    private final double objectiveChangeTolerance;
//    private final double gradientTolerance;
//    private final boolean checkForPointConvergence;
//    private final boolean checkForObjectiveConvergence;
//    private final boolean checkForGradientConvergence;
//    private final Function<Vector, Boolean> additionalCustomConvergenceCriterion;
//    private final LineSearch lineSearch;
//    private final QuasiNewtonSolver.Method method;
//    private final int m;
//    private final double symmetricRankOneSkippingParameter;
//
//    /**
//     * This abstract class needs to be extended by the builder of its parent binary logistic regression class. It
//     * provides an implementation for those parts of those builders that are common. This is basically part of a small
//     * "hack" so that we can have inheritable builder classes.
//     *
//     * @param   <T> This type corresponds to the type of the final object to be built. That is, the super class of the
//     *              builder class that extends this class, which in this case will be the
//     *              {@link LogisticRegressionQuasiNewton} class.
//     */
//    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
//            extends AbstractTrainableLogisticRegression.AbstractBuilder<T> {
//        protected int maximumNumberOfIterations = 10000;
//        protected int maximumNumberOfFunctionEvaluations = 1000000;
//        protected double pointChangeTolerance = 1e-10;
//        protected double objectiveChangeTolerance = 1e-10;
//        protected double gradientTolerance = 1e-6;
//        protected boolean checkForPointConvergence = true;
//        protected boolean checkForObjectiveConvergence = true;
//        protected boolean checkForGradientConvergence = true;
//        protected Function<Vector, Boolean> additionalCustomConvergenceCriterion = currentPoint -> false;
//        protected LineSearch lineSearch = null;
//        protected QuasiNewtonSolver.Method method = QuasiNewtonSolver.Method.BROYDEN_FLETCHER_GOLDFARB_SHANNO;
//        protected int m = 1;
//        protected double symmetricRankOneSkippingParameter = 1e-8;
//
//        protected AbstractBuilder(int numberOfFeatures) {
//            super(numberOfFeatures);
//        }
//
//        protected AbstractBuilder(int numberOfFeatures, Vector weights) {
//            super(numberOfFeatures, weights);
//        }
//
//        public T maximumNumberOfIterations(int maximumNumberOfIterations) {
//            this.maximumNumberOfIterations = maximumNumberOfIterations;
//            return self();
//        }
//
//        public T maximumNumberOfFunctionEvaluations(int maximumNumberOfFunctionEvaluations) {
//            this.maximumNumberOfFunctionEvaluations = maximumNumberOfFunctionEvaluations;
//            return self();
//        }
//
//        public T pointChangeTolerance(double pointChangeTolerance) {
//            this.pointChangeTolerance = pointChangeTolerance;
//            return self();
//        }
//
//        public T objectiveChangeTolerance(double objectiveChangeTolerance) {
//            this.objectiveChangeTolerance = objectiveChangeTolerance;
//            return self();
//        }
//
//        public T gradientTolerance(double gradientTolerance) {
//            this.gradientTolerance = gradientTolerance;
//            return self();
//        }
//
//        public T checkForPointConvergence(boolean checkForPointConvergence) {
//            this.checkForPointConvergence = checkForPointConvergence;
//            return self();
//        }
//
//        public T checkForObjectiveConvergence(boolean checkForObjectiveConvergence) {
//            this.checkForObjectiveConvergence = checkForObjectiveConvergence;
//            return self();
//        }
//
//        public T checkForGradientConvergence(boolean checkForGradientConvergence) {
//            this.checkForGradientConvergence = checkForGradientConvergence;
//            return self();
//        }
//
//        public T additionalCustomConvergenceCriterion(Function<Vector, Boolean> additionalCustomConvergenceCriterion) {
//            this.additionalCustomConvergenceCriterion = additionalCustomConvergenceCriterion;
//            return self();
//        }
//
//        public T lineSearch(LineSearch lineSearch) {
//            this.lineSearch = lineSearch;
//            return self();
//        }
//
//        public T method(QuasiNewtonSolver.Method method) {
//            this.method = method;
//            return self();
//        }
//
//        public T m(int m) {
//            this.m = m;
//            return self();
//        }
//
//        public T symmetricRankOneSkippingParameter(double symmetricRankOneSkippingParameter) {
//            this.symmetricRankOneSkippingParameter = symmetricRankOneSkippingParameter;
//            return self();
//        }
//
//        @Override
//        public T setParameter(String name, Object computeValue) {
//            switch (name) {
//                case "maximumNumberOfIterations":
//                    maximumNumberOfIterations = (int) computeValue;
//                    break;
//                case "maximumNumberOfFunctionEvaluations":
//                    maximumNumberOfFunctionEvaluations = (int) computeValue;
//                    break;
//                case "pointChangeTolerance":
//                    pointChangeTolerance = (double) computeValue;
//                    break;
//                case "objectiveChangeTolerance":
//                    objectiveChangeTolerance = (double) computeValue;
//                    break;
//                case "gradientTolerance":
//                    gradientTolerance = (double) computeValue;
//                    break;
//                case "checkForPointConvergence":
//                    checkForPointConvergence = (boolean) computeValue;
//                    break;
//                case "checkForObjectiveConvergence":
//                    checkForObjectiveConvergence = (boolean) computeValue;
//                    break;
//                case "checkForGradientConvergence":
//                    checkForGradientConvergence = (boolean) computeValue;
//                    break;
//                case "method":
//                    method = QuasiNewtonSolver.Method.valueOf((String) computeValue);
//                    break;
//                case "m":
//                    m = (int) computeValue;
//                    break;
//                case "symmetricRankOneSkippingParameter":
//                    symmetricRankOneSkippingParameter = (double) computeValue;
//                    break;
//                default:
//                    super.setParameter(name, computeValue);
//            }
//            return self();
//        }
//
//        @Override
//        public LogisticRegressionQuasiNewton build() {
//            return new LogisticRegressionQuasiNewton(this);
//        }
//    }
//
//    /**
//     * The builder class for this class. This is basically part of a small "hack" so that we can have inheritable
//     * builder classes.
//     */
//    public static class Builder extends AbstractBuilder<Builder>
//            implements TrainableClassifier.Builder<Vector, Double> {
//        /**
//         * Constructs a builder object for a binary logistic regression model that will be trained with the provided
//         * training data using the stochastic gradient descent algorithm.
//         *
//         * @param   numberOfFeatures    The number of features used.
//         */
//        public Builder(int numberOfFeatures) {
//            super(numberOfFeatures);
//        }
//
//        public Builder(int numberOfFeatures, Vector weights) {
//            super(numberOfFeatures, weights);
//        }
//
//        /** {@inheritDoc} */
//        @Override
//        protected Builder self() {
//            return this;
//        }
//    }
//
//    /**
//     * Constructs a binary logistic regression object that uses the adaptive gradient algorithm to train the model,
//     * given an appropriate builder object. This constructor can only be used from within the builder class of this
//     * class.
//     *
//     * @param   builder The builder object to use.
//     */
//    private LogisticRegressionQuasiNewton(AbstractBuilder<?> builder) {
//        super(builder);
//
//        maximumNumberOfIterations = builder.maximumNumberOfIterations;
//        maximumNumberOfFunctionEvaluations = builder.maximumNumberOfFunctionEvaluations;
//        pointChangeTolerance = builder.pointChangeTolerance;
//        objectiveChangeTolerance = builder.objectiveChangeTolerance;
//        gradientTolerance = builder.gradientTolerance;
//        checkForPointConvergence = builder.checkForPointConvergence;
//        checkForObjectiveConvergence = builder.checkForObjectiveConvergence;
//        checkForGradientConvergence = builder.checkForGradientConvergence;
//        additionalCustomConvergenceCriterion = builder.additionalCustomConvergenceCriterion;
//        lineSearch = builder.lineSearch;
//        method = builder.method;
//        m = builder.m;
//        symmetricRankOneSkippingParameter = builder.symmetricRankOneSkippingParameter;
//    }
//
//    @Override
//    public ClassifierType type() {
//        return ClassifierType.LOGISTIC_REGRESSION_QUASI_NEWTON;
//    }
//
//    /** {@inheritDoc} */
//    @Override
//    protected void train() {
//        weights =  new QuasiNewtonSolver.Builder(new LikelihoodFunction(), weights)
//                .maximumNumberOfIterations(maximumNumberOfIterations)
//                .maximumNumberOfFunctionEvaluations(maximumNumberOfFunctionEvaluations)
//                .pointChangeTolerance(pointChangeTolerance)
//                .objectiveChangeTolerance(objectiveChangeTolerance)
//                .gradientTolerance(gradientTolerance)
//                .checkForPointConvergence(checkForPointConvergence)
//                .checkForObjectiveConvergence(checkForObjectiveConvergence)
//                .checkForGradientConvergence(checkForGradientConvergence)
//                .additionalCustomConvergenceCriterion(additionalCustomConvergenceCriterion)
//                .lineSearch(lineSearch)
//                .method(method)
//                .m(m)
//                .symmetricRankOneSkippingParameter(symmetricRankOneSkippingParameter)
//                // TODO: Add support for regularization to the corresponding solvers.
////                .useL1Regularization(useL1Regularization)
////                .l1RegularizationWeight(l1RegularizationWeight)
////                .useL2Regularization(useL2Regularization)
////                .l2RegularizationWeight(l2RegularizationWeight)
//                .loggingLevel(loggingLevel)
//                .build()
//                .solve();
//    }
//
//    /** {@inheritDoc} */
//    @Override
//    public void write(OutputStream outputStream, boolean includeType) throws IOException {
//        super.write(outputStream, includeType);
//
//        UnsafeSerializationUtilities.writeInt(outputStream, maximumNumberOfIterations);
//        UnsafeSerializationUtilities.writeInt(outputStream, maximumNumberOfFunctionEvaluations);
//        UnsafeSerializationUtilities.writeDouble(outputStream, pointChangeTolerance);
//        UnsafeSerializationUtilities.writeDouble(outputStream, objectiveChangeTolerance);
//        UnsafeSerializationUtilities.writeDouble(outputStream, gradientTolerance);
//        UnsafeSerializationUtilities.writeBoolean(outputStream, checkForPointConvergence);
//        UnsafeSerializationUtilities.writeBoolean(outputStream, checkForObjectiveConvergence);
//        UnsafeSerializationUtilities.writeBoolean(outputStream, checkForGradientConvergence);
//        UnsafeSerializationUtilities.writeInt(outputStream, method.ordinal());
//        UnsafeSerializationUtilities.writeInt(outputStream, m);
//        UnsafeSerializationUtilities.writeDouble(outputStream, symmetricRankOneSkippingParameter);
//    }
//
//    public static LogisticRegressionQuasiNewton read(InputStream inputStream, boolean includeType) throws IOException {
//        if (includeType) {
//            ClassifierType classifierType = ClassifierType.values()[UnsafeSerializationUtilities.readInt(inputStream)];
//            if (!ClassifierType.LOGISTIC_REGRESSION_ADAGRAD
//                    .getStorageCompatibleTypes()
//                    .contains(classifierType))
//                throw new InvalidObjectException("The stored classifier is of type " + classifierType.name() + "!");
//        }
//        int numberOfFeatures = UnsafeSerializationUtilities.readInt(inputStream);
//        boolean sparse = UnsafeSerializationUtilities.readBoolean(inputStream);
//        Vector weights = Vectors.build(inputStream);
//        double l1RegularizationWeight = UnsafeSerializationUtilities.readDouble(inputStream);
//        double l2RegularizationWeight = UnsafeSerializationUtilities.readDouble(inputStream);
//        int loggingLevel = UnsafeSerializationUtilities.readInt(inputStream);
//        int maximumNumberOfIterations = UnsafeSerializationUtilities.readInt(inputStream);
//        int maximumNumberOfFunctionEvaluations = UnsafeSerializationUtilities.readInt(inputStream);
//        double pointChangeTolerance = UnsafeSerializationUtilities.readDouble(inputStream);
//        double objectiveChangeTolerance = UnsafeSerializationUtilities.readDouble(inputStream);
//        double gradientTolerance = UnsafeSerializationUtilities.readDouble(inputStream);
//        boolean checkForPointConvergence = UnsafeSerializationUtilities.readBoolean(inputStream);
//        boolean checkForObjectiveConvergence = UnsafeSerializationUtilities.readBoolean(inputStream);
//        boolean checkForGradientConvergence = UnsafeSerializationUtilities.readBoolean(inputStream);
//        QuasiNewtonSolver.Method method =
//                QuasiNewtonSolver.Method.values()[UnsafeSerializationUtilities.readInt(inputStream)];
//        int m = UnsafeSerializationUtilities.readInt(inputStream);
//        double symmetricRankOneSkippingParameter = UnsafeSerializationUtilities.readDouble(inputStream);
//        return new Builder(numberOfFeatures, weights)
//                .sparse(sparse)
//                .l1RegularizationWeight(l1RegularizationWeight)
//                .l2RegularizationWeight(l2RegularizationWeight)
//                .loggingLevel(loggingLevel)
//                .maximumNumberOfIterations(maximumNumberOfIterations)
//                .maximumNumberOfFunctionEvaluations(maximumNumberOfFunctionEvaluations)
//                .pointChangeTolerance(pointChangeTolerance)
//                .objectiveChangeTolerance(objectiveChangeTolerance)
//                .gradientTolerance(gradientTolerance)
//                .checkForPointConvergence(checkForPointConvergence)
//                .checkForObjectiveConvergence(checkForObjectiveConvergence)
//                .checkForGradientConvergence(checkForGradientConvergence)
//                .method(method)
//                .m(m)
//                .symmetricRankOneSkippingParameter(symmetricRankOneSkippingParameter)
//                .build();
//    }
//}
