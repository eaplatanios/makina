//package org.platanios.learn.classification;
//
//import org.platanios.learn.data.PredictedDataInstance;
//import org.platanios.learn.kernel.KernelFunction;
//import org.platanios.learn.kernel.LinearKernelFunction;
//import org.platanios.math.matrix.Vector;
//import org.platanios.math.matrix.VectorType;
//import org.platanios.math.matrix.Vectors;
//import org.platanios.utilities.UnsafeSerializationUtilities;
//
//import java.io.IOException;
//import java.io.InputStream;
//import java.io.InvalidObjectException;
//import java.io.OutputStream;
//
///**
// * @author Emmanouil Antonios Platanios
// */
//public class SupportVectorMachinePrediction implements Classifier<Vector, Boolean> {
//    /** The number of features used. */
//    protected int numberOfFeatures;
//    /** Indicates whether sparse vectors are being used or not. */
//    protected boolean sparse;
//    /** Indicates whether a separate bias term must be used along with the feature weights. Note that if a computeValue of
//     * 1 has already been appended to all feature vectors, then there is no need for a bias term. */
//    protected boolean useBiasTerm;
//    /** The kernel function to use. */
//    protected KernelFunction<Vector> kernelFunction;
//
//    /** The weights (i.e., parameters) used by this support vector machine model. Note that the size of this vector is
//     * equal to 1 + {@link #numberOfFeatures}. */
//    protected Vector weights;
//
//    /**
//     * This abstract class needs to be extended by the builders of all support vector machine classes. It provides
//     * an implementation for those parts of those builders that are common. This is basically part of a small "hack" so
//     * that we can have inheritable builder classes.
//     *
//     * @param   <T> This type corresponds to the type of the final object to be built. That is, the super class of the
//     *              builder class that extends this class.
//     */
//    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>> {
//        /** A self-reference to this builder class. This is basically part of a small "hack" so that we can have
//         * inheritable builder classes. */
//        protected abstract T self();
//
//        /** The number of features used. */
//        protected int numberOfFeatures;
//        /** Indicates whether sparse vectors should be used or not. */
//        private boolean sparse = false;
//        /** Indicates whether a separate bias term must be used along with the feature weights. Note that if a computeValue of
//         * 1 has already been appended to all feature vectors, then there is no need for a bias term. */
//        protected boolean useBiasTerm = true;
//        /** The kernel function to use. */
//        protected KernelFunction<Vector> kernelFunction = new LinearKernelFunction();
//        /** The weights (i.e., parameters) used by this support vector machine model. Note that the size of this vector
//         * is equal to 1 + {@link #numberOfFeatures}. */
//        protected Vector weights = null;
//
//        protected AbstractBuilder() { }
//
//        protected AbstractBuilder(int numberOfFeatures, Vector weights) {
//            this.numberOfFeatures = numberOfFeatures;
//            this.weights = weights;
//        }
//
//        /**
//         * Sets the {@link #sparse} field that indicates whether sparse vectors should be used.
//         *
//         * @param   sparse  The computeValue to which to set the {@link #sparse} field.
//         * @return          This builder object itself. That is done so that we can use a nice and expressive code
//         *                  format when we build objects using this builder class.
//         */
//        public T sparse(boolean sparse) {
//            this.sparse = sparse;
//            return self();
//        }
//
//        /**
//         * Sets the {@link #useBiasTerm} field that indicates whether a separate bias term must be used along with the
//         * feature weights. Note that if a computeValue of 1 has already been appended to all feature vectors, then there is no
//         * need for a bias term.
//         *
//         * @param   useBiasTerm The computeValue to which to set the {@link #useBiasTerm} field.
//         * @return              This builder object itself. That is done so that we can use a nice and expressive code
//         *                      format when we build objects using this builder class.
//         */
//        public T useBiasTerm(boolean useBiasTerm) {
//            this.useBiasTerm = useBiasTerm;
//            return self();
//        }
//
//        /**
//         * Sets the {@link #kernelFunction} field that indicates which kernel function will be used by the support
//         * vector machine being built. The default kernel function is a linear kernel function with no shift (i.e., an
//         * inner product).
//         *
//         * @param   kernelFunction  The computeValue to which to set the {@link #kernelFunction} field.
//         * @return                  This builder object itself. That is done so that we can use a nice and expressive
//         *                          code format when we build objects using this builder class.
//         */
//        public T kernelFunction(KernelFunction<Vector> kernelFunction) {
//            this.kernelFunction = kernelFunction;
//            return self();
//        }
//
//        public T setParameter(String name, Object computeValue) {
//            switch (name) {
//                case "sparse":
//                    sparse = (boolean) computeValue;
//                    break;
//                case "useBiasTerm":
//                    useBiasTerm = (boolean) computeValue;
//                    break;
//                default:
//                    break;
//            }
//            return self();
//        }
//
//        public SupportVectorMachinePrediction build() {
//            return new SupportVectorMachinePrediction(this);
//        }
//    }
//
//    /**
//     * The builder class for this abstract class. This is basically part of a small "hack" so that we can have
//     * inheritable builder classes.
//     */
//    public static class Builder extends AbstractBuilder<Builder> {
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
//     * Constructs a support vector machine object given an appropriate builder object. This constructor can only be
//     * used from within the builder class of this class.
//     *
//     * @param   builder The builder object to use.
//     */
//    protected SupportVectorMachinePrediction(AbstractBuilder<?> builder) {
//        numberOfFeatures = builder.numberOfFeatures;
//        sparse = builder.sparse;
//        useBiasTerm = builder.useBiasTerm;
//        kernelFunction = builder.kernelFunction;
//        if (builder.weights != null)
//            weights = builder.weights;
//        else
//        if (builder.sparse)
//            weights = Vectors.build(useBiasTerm ? numberOfFeatures + 1 : numberOfFeatures, VectorType.SPARSE);
//        else
//            weights = Vectors.build(useBiasTerm ? numberOfFeatures + 1 : numberOfFeatures, VectorType.DENSE);
//    }
//
//    @Override
//    public ClassifierType type() {
//        return ClassifierType.SUPPORT_VECTOR_MACHINE_PREDICTION;
//    }
//
//    @Override
//    public PredictedDataInstance<Vector, Boolean> predictInPlace(PredictedDataInstance<Vector, Boolean> dataInstance) {
//        double distanceFromSeparator = kernelFunction.getValue(weights, dataInstance.features());
//        dataInstance.probability(distanceFromSeparator);
//        if (distanceFromSeparator >= 0)
//            dataInstance.label(true);
//        else
//            dataInstance.label(false);
//        return dataInstance;
//    }
//
//    @Override
//    public void write(OutputStream outputStream, boolean includeType) throws IOException {
//        if (includeType)
//            UnsafeSerializationUtilities.writeInt(outputStream, type().ordinal());
//        UnsafeSerializationUtilities.writeInt(outputStream, numberOfFeatures);
//        UnsafeSerializationUtilities.writeBoolean(outputStream, sparse);
//        weights.write(outputStream, true);
//    }
//
//    public static SupportVectorMachinePrediction read(InputStream inputStream, boolean includeType) throws IOException {
//        if (includeType) {
//            ClassifierType classifierType = ClassifierType.values()[UnsafeSerializationUtilities.readInt(inputStream)];
//            if (!ClassifierType.SUPPORT_VECTOR_MACHINE_PREDICTION
//                    .getStorageCompatibleTypes()
//                    .contains(classifierType))
//                throw new InvalidObjectException("The stored classifier is of type " + classifierType.name() + "!");
//        }
//        int numberOfFeatures = UnsafeSerializationUtilities.readInt(inputStream);
//        boolean sparse = UnsafeSerializationUtilities.readBoolean(inputStream);
//        Vector weights = Vectors.build(inputStream);
//        return new Builder(numberOfFeatures, weights).sparse(sparse).build();
//    }
//}
