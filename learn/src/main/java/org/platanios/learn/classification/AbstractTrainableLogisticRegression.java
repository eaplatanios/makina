package org.platanios.learn.classification;

import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.LabeledDataInstance;
import org.platanios.learn.math.matrix.SparseVector;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.AbstractStochasticFunctionUsingDataSet;
import org.platanios.learn.serialization.UnsafeSerializationUtilities;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * This class implements a binary logistic regression model that is trained using the stochastic gradient descent
 * algorithm.
 *
 * @author Emmanouil Antonios Platanios
 */
abstract class AbstractTrainableLogisticRegression
        extends LogisticRegressionPrediction implements TrainableClassifier<Vector, Double> {
    /** The data used to train this model. */
    protected DataSet<LabeledDataInstance<Vector, Double>> trainingDataSet;
    protected double l1RegularizationWeight;
    protected double l2RegularizationWeight;
    protected int loggingLevel;

    /**
     * This abstract class needs to be extended by the builder of its parent binary logistic regression class. It
     * provides an implementation for those parts of those builders that are common. This is basically part of a small
     * "hack" so that we can have inheritable builder classes.
     *
     * @param   <T> This type corresponds to the type of the final object to be built. That is, the super class of the
     *              builder class that extends this class.
     */
    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends LogisticRegressionPrediction.AbstractBuilder<T> {
        protected double l1RegularizationWeight = 0.0;
        protected double l2RegularizationWeight = 0.0;
        protected int loggingLevel = 0;

        /**
         * Constructs a builder object for a binary logistic regression model that will be trained with the provided
         * training data. This constructor should be used if the logistic regression model that is being built is going
         * to be trained.
         *
         * @param   numberOfFeatures    The number of features used.
         */
        protected AbstractBuilder(int numberOfFeatures) {
            this.numberOfFeatures = numberOfFeatures;
        }

        protected AbstractBuilder(int numberOfFeatures, Vector weights) {
            super(numberOfFeatures, weights);
        }

        /**
         * Sets the {@link #l1RegularizationWeight} field that contains the computeValue of the /(L_1/) regularization weight
         * used. This variable is only used when {@link #useL1Regularization} is set to true.
         *
         * @param   l1RegularizationWeight  The computeValue to which to set the {@link #l1RegularizationWeight} field.
         * @return                          This builder object itself. That is done so that we can use a nice and
         *                                  expressive code format when we build objects using this builder class.
         */
        public T l1RegularizationWeight(double l1RegularizationWeight) {
            this.l1RegularizationWeight = l1RegularizationWeight;
            return self();
        }

        /**
         * Sets the {@link #l2RegularizationWeight} field that contains the computeValue of the /(L_2/) regularization weight
         * used. This variable is only used when {@link #useL2Regularization} is set to true.
         *
         * @param   l2RegularizationWeight  The computeValue to which to set the {@link #l2RegularizationWeight} field.
         * @return                          This builder object itself. That is done so that we can use a nice and
         *                                  expressive code format when we build objects using this builder class.
         */
        public T l2RegularizationWeight(double l2RegularizationWeight) {
            this.l2RegularizationWeight = l2RegularizationWeight;
            return self();
        }

        public T loggingLevel(int loggingLevel) {
            this.loggingLevel = loggingLevel;
            return self();
        }

        @Override
        public T setParameter(String name, Object value) {
            switch (name) {
                case "l1RegularizationWeight":
                    l1RegularizationWeight = (double) value;
                    break;
                case "l2RegularizationWeight":
                    l2RegularizationWeight = (double) value;
                    break;
                default:
                    super.setParameter(name, value);
            }
            return self();
        }
    }

    /**
     * The builder class for this class. This is basically part of a small "hack" so that we can have inheritable
     * builder classes.
     */
    public static class Builder extends AbstractBuilder<Builder> {
        /**
         * Constructs a builder object for a binary logistic regression model that will be trained with the provided
         * training data.
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
     * Constructs a binary logistic regression object, given an appropriate builder object. This constructor can only be
     * used from within the builder class of this class and from its subclasses.
     *
     * @param   builder The builder object to use.
     */
    protected AbstractTrainableLogisticRegression(AbstractBuilder<?> builder) {
        super(builder);

        l1RegularizationWeight = builder.l1RegularizationWeight;
        l2RegularizationWeight = builder.l2RegularizationWeight;
        loggingLevel = builder.loggingLevel;
    }

    @Override
    @SuppressWarnings("unchecked")
    public boolean train(DataSet<? extends LabeledDataInstance<Vector, Double>> trainingDataSet) {
        this.trainingDataSet = (DataSet<LabeledDataInstance<Vector, Double>>) trainingDataSet;
        try {
            train();
            if (sparse)
                ((SparseVector) weights).compact();
            return true;
        } catch(Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Trains this logistic regression model.
     */
    protected abstract void train();

    /**
     * Class implementing the likelihood function for the binary logistic regression model. No function is provided to
     * compute the hessian matrix because no binary logistic regression class has to use it.
     *
     * //TODO: Add Hessian method implementation.
     */
    protected class LikelihoodFunction extends AbstractFunction {
        private Iterator<LabeledDataInstance<Vector, Double>> trainingDataIterator;

        public LikelihoodFunction() {
            trainingDataIterator = trainingDataSet.iterator();
        }

        @Override
        public boolean equals(Object other) {
            // should we iterator the trainingDataSet here? instead of just checking identity?
            if (other == this) {
                return true;
            }

            return false;
        }

        @Override
        public int hashCode() {
            return System.identityHashCode(this);
        }

        /**
         * Computes the computeValue of the likelihood function for the binary logistic regression model.
         *
         * @param   weights The current weights vector.
         * @return          The computeValue of the logistic regression likelihood function.
         */
        @Override
        public double computeValue(Vector weights) {
            double likelihood = 0;
            while (trainingDataIterator.hasNext()) {
                LabeledDataInstance<Vector, Double> dataInstance = trainingDataIterator.next();
                double probability = useBiasTerm ?
                        weights.dotPlusConstant(dataInstance.features()) :
                        weights.dot(dataInstance.features());
                likelihood += probability * (dataInstance.label() - 1) - Math.log(1 + Math.exp(-probability));
            }
            return -likelihood;
        }

        /**
         * Computes the gradient of the likelihood function for the binary logistic regression model.
         *
         * @param   weights The current weights vector.
         * @return          The gradient vector of the logistic regression likelihood function.
         */
        @Override
        public Vector computeGradient(Vector weights) {
            Vector gradient = Vectors.build(weights.size(), weights.type());
            while (trainingDataIterator.hasNext()) {
                LabeledDataInstance<Vector, Double> dataInstance = trainingDataIterator.next();
                if (useBiasTerm)
                    gradient.saxpyPlusConstantInPlace(
                            (1 / (1 + Math.exp(-weights.dotPlusConstant(dataInstance.features())))) - dataInstance.label(),
                            dataInstance.features()
                    );
                else
                    gradient.saxpyInPlace(
                            (1 / (1 + Math.exp(-weights.dot(dataInstance.features())))) - dataInstance.label(),
                            dataInstance.features()
                    );
            }
            return gradient;
        }
    }

    /**
     * Class implementing the likelihood function for the binary logistic regression model for use with stochastic
     * optimization solvers.
     */
    protected class StochasticLikelihoodFunction
            extends AbstractStochasticFunctionUsingDataSet<LabeledDataInstance<Vector, Double>> {
        public StochasticLikelihoodFunction() {
        	this.dataSet = trainingDataSet;
        }
        
        public StochasticLikelihoodFunction(Random random) {
        	this();
        	this.random = random;
        }

        /**
         * Computes the gradient of the likelihood function for the binary logistic regression model.
         *
         * @param   weights     The current weights vector.
         * @param   dataBatch
         * @return              The gradient vector of the logistic regression likelihood function.
         */
        @Override
        public Vector estimateGradient(Vector weights, List<LabeledDataInstance<Vector, Double>> dataBatch) {
            Vector gradient = Vectors.build(weights.size(), weights.type());
            for (LabeledDataInstance<Vector, Double> dataInstance : dataBatch) {
                if (useBiasTerm)
                    gradient.saxpyPlusConstantInPlace(
                            (1 / (1 + Math.exp(-weights.dotPlusConstant(dataInstance.features())))) - dataInstance.label(),
                            dataInstance.features()
                    );
                else
                    gradient.saxpyInPlace(
                            (1 / (1 + Math.exp(-weights.dot(dataInstance.features())))) - dataInstance.label(),
                            dataInstance.features()
                    );
            }
            return gradient;
        }
    }

    /** {@inheritDoc} */
    @Override
    public void write(OutputStream outputStream, boolean includeType) throws IOException {
        super.write(outputStream, includeType);

        UnsafeSerializationUtilities.writeDouble(outputStream, l1RegularizationWeight);
        UnsafeSerializationUtilities.writeDouble(outputStream, l2RegularizationWeight);
        UnsafeSerializationUtilities.writeInt(outputStream, loggingLevel);
    }
}
