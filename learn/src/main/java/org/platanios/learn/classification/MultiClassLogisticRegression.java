package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.*;
import org.platanios.learn.optimization.QuasiNewtonSolver;
import org.platanios.learn.optimization.Solver;
import org.platanios.learn.optimization.StochasticGradientDescentSolver;
import org.platanios.learn.optimization.StochasticSolverStepSize;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.AbstractStochasticFunctionUsingList;

import java.util.Arrays;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class MultiClassLogisticRegression {
    private final Solver solver;
    private final List<TrainingData.Entry> trainingData;
    private final int trainingDataSize;
    private final int numberOfFeatures;
    private final int numberOfClasses;

    private Matrix weights;

    public static class Builder {
        private final List<TrainingData.Entry> trainingData;
        private final int trainingDataSize;
        private final int numberOfFeatures;
        private final int numberOfClasses;

        private boolean stochastic = false;
        private boolean largeScale = false;

        public Builder(TrainingData trainingData) {
            this.trainingData = trainingData.getData();
            Integer[] trainingDataLabels = new Integer[this.trainingData.size()];
            for (int i = 0; i < this.trainingData.size(); i++) {
                trainingDataLabels[i] = this.trainingData.get(i).label;
            }
            trainingDataSize = this.trainingData.size();
            numberOfClasses = 1 + Arrays.asList(trainingDataLabels).stream().max(Integer::compare).get();
            numberOfFeatures = this.trainingData.get(0).features.size();
        }

        public Builder stochastic(boolean stochastic) {
            this.stochastic = stochastic;
            return this;
        }

        public Builder largeScale(boolean largeScale) {
            this.largeScale = largeScale;
            return this;
        }

        public MultiClassLogisticRegression build() {
            return new MultiClassLogisticRegression(this);
        }
    }

    private MultiClassLogisticRegression(Builder builder) {
        trainingData = builder.trainingData;
        trainingDataSize = builder.trainingDataSize;
        numberOfFeatures = builder.numberOfFeatures;
        numberOfClasses = builder.numberOfClasses;
        weights = new Matrix(numberOfFeatures, numberOfClasses - 1);

        if (builder.stochastic) {
            solver = new StochasticGradientDescentSolver.Builder(new StochasticLikelihoodFunction(),
                                                                 Vectors.dense(weights.getColumnPackedArrayCopy()))
                    .sampleWithReplacement(false)
                    .maximumNumberOfIterations(10000)
                    .maximumNumberOfIterationsWithNoPointChange(5)
                    .pointChangeTolerance(1e-10)
                    .checkForPointConvergence(true)
                    .batchSize(10)
                    .stepSize(StochasticSolverStepSize.SCALED)
                    .stepSizeParameters(10, 0.75)
                    .build();
            return;
        }
        if (!builder.largeScale) {
            solver = new QuasiNewtonSolver.Builder(new LikelihoodFunction(),
                                                   Vectors.dense(weights.getColumnPackedArrayCopy()))
                    .method(QuasiNewtonSolver.Method.BROYDEN_FLETCHER_GOLDFARB_SHANNO)
                    .maximumNumberOfIterations(10000)
                    .maximumNumberOfFunctionEvaluations(1000000)
                    .pointChangeTolerance(1e-10)
                    .objectiveChangeTolerance(1e-10)
                    .gradientTolerance(1e-6)
                    .checkForPointConvergence(true)
                    .checkForObjectiveConvergence(true)
                    .checkForGradientConvergence(true)
                    .build();
        } else {
            solver = new QuasiNewtonSolver.Builder(new LikelihoodFunction(),
                                                   Vectors.dense(weights.getColumnPackedArrayCopy()))
                    .method(QuasiNewtonSolver.Method.LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO)
                    .m(10)
                    .maximumNumberOfIterations(10000)
                    .maximumNumberOfFunctionEvaluations(1000000)
                    .pointChangeTolerance(1e-10)
                    .objectiveChangeTolerance(1e-10)
                    .gradientTolerance(1e-6)
                    .checkForPointConvergence(true)
                    .checkForObjectiveConvergence(true)
                    .checkForGradientConvergence(true)
                    .build();
        }
    }

    public void train() {
        weights = new Matrix(numberOfFeatures, numberOfClasses);
        weights.setSubMatrix(
                0,
                weights.getRowDimension() - 1,
                0,
                numberOfClasses - 2,
                new Matrix(solver.solve(), numberOfFeatures)
        );
    }

    public double[] predict(double[] point) {
        Vector probabilities = weights.transpose().multiply(Vectors.dense(point));
        probabilities = probabilities.sub(MatrixUtilities.computeLogSumExp(probabilities));
        probabilities = probabilities.map(Math::exp);
        return probabilities.getDenseArray();
    }

    public double[][] predict(double[][] points) {
        double[][] probabilities = new double[points.length][];
        for (int i = 0; i < points.length; i++) {
            probabilities[i] = predict(points[i]);
        }
        return probabilities;
    }

    /**
     * Class implementing the likelihood function for the multi-class logistic regression model.
     */
    private class LikelihoodFunction extends AbstractFunction {

        @Override
        public boolean equals(Object other) {
            return other == this;
        }

        @Override
        public int hashCode() {
            return System.identityHashCode(this);
        }

        /**
         * Computes the computeValue of the likelihood function for the multi-class logistic regression model.
         *
         * @param   weights The current weights vector.
         * @return          The computeValue of the logistic regression likelihood function.
         */
        @Override
        public double computeValue(Vector weights) {
            Matrix W = new Matrix(weights, numberOfFeatures);
            double likelihood = 0;
            for (int n = 0; n < trainingDataSize; n++) {
                Vector innerProduct = W.transpose().multiply(trainingData.get(n).features);
                if (trainingData.get(n).label != numberOfClasses - 1) {
                    likelihood += innerProduct.get(trainingData.get(n).label);
                }
                Vector innerProductWithLastClass = Vectors.build(numberOfClasses, VectorType.DENSE);
                innerProductWithLastClass.set(0, numberOfClasses - 2, innerProduct);
                likelihood -= MatrixUtilities.computeLogSumExp(innerProductWithLastClass);
            }
            return -likelihood;
        }

        /**
         * Computes the gradient of the likelihood function for the multi-class logistic regression model.
         *
         * @param   weights The current weights vector.
         * @return          The gradient vector of the logistic regression likelihood function.
         */
        @Override
        public Vector computeGradient(Vector weights) {
            Matrix W = new Matrix(weights, numberOfFeatures);
            Matrix gradient = new Matrix(W.getRowDimension(), W.getColumnDimension());
            for (int n = 0; n < trainingDataSize; n++) {
                Vector probabilities = W.transpose().multiply(trainingData.get(n).features);
                Vector innerProductWithLastClass = Vectors.build(numberOfClasses, VectorType.DENSE);
                innerProductWithLastClass.set(0, numberOfClasses - 2, probabilities);
                probabilities = probabilities.sub(MatrixUtilities.computeLogSumExp(innerProductWithLastClass));
                probabilities = probabilities.map(Math::exp);
                if (trainingData.get(n).label != numberOfClasses - 1) {
                    probabilities.set(trainingData.get(n).label, probabilities.get(trainingData.get(n).label) - 1);
                }
                for (int c = 0; c < numberOfClasses - 1; c++) {
                    gradient.setColumn(
                            c,
                            gradient.getColumn(c).add(trainingData.get(n).features.mult(probabilities.get(c)))
                    );
                }
            }
            return Vectors.dense(gradient.getColumnPackedArrayCopy());
        }

        /**
         * Computes the Hessian matrix of the likelihood function for the multi-class logistic regression model. This
         * method is heavy computationally and quasi-Newton methods seem to perform better in practice (with other
         * methods that use the Hessian matrix directly, usually its inverse has to be computed as well, which is also a
         * heavy operation computationally). That is why this method is not used in the current implementation of the
         * logistic regression algorithm; quasi-Newton methods are used instead that approximate the inverse of the
         * Hessian matrix directly.
         *
         * @param   weights The current weights vector.
         * @return          The Hessian matrix of the logistic regression likelihood function.
         */
        @Override
        public Matrix computeHessian(Vector weights) {
            Matrix W = new Matrix(weights, numberOfFeatures);
            Matrix hessian = new Matrix(new double[weights.size()][weights.size()]);
            for (int n = 0; n < trainingDataSize; n++) {
                Vector probabilities = W.transpose().multiply(trainingData.get(n).features);
                Vector innerProductWithLastClass = Vectors.build(numberOfClasses, VectorType.DENSE);
                innerProductWithLastClass.set(0, numberOfClasses - 2, probabilities);
                probabilities = probabilities.sub(MatrixUtilities.computeLogSumExp(innerProductWithLastClass));
                probabilities = probabilities.map(Math::exp);
                for (int i = 0; i < numberOfClasses - 1; i++) {
                    for (int j = 0; j <= i; j++) {
                        Matrix subMatrix = hessian.getSubMatrix(i * numberOfFeatures,
                                                                (i + 1) * numberOfFeatures - 1,
                                                                j * numberOfFeatures,
                                                                (j + 1) * numberOfFeatures - 1);
                        if (i != j) {
                            subMatrix = subMatrix.add(
                                    trainingData.get(n).features
                                            .outer(trainingData.get(n).features)
                                            .multiply(-probabilities.get(i) * probabilities.get(j))
                            );
                            hessian.setSubMatrix(
                                    i * numberOfFeatures,
                                    (i + 1) * numberOfFeatures - 1,
                                    j * numberOfFeatures,
                                    (j + 1) * numberOfFeatures - 1,
                                    subMatrix
                            );
                            hessian.setSubMatrix(
                                    j * numberOfFeatures,
                                    (j + 1) * numberOfFeatures - 1,
                                    i * numberOfFeatures,
                                    (i + 1) * numberOfFeatures - 1,
                                    subMatrix
                            );
                        } else {
                            subMatrix = subMatrix.add(
                                    trainingData.get(n).features
                                            .outer(trainingData.get(n).features)
                                            .multiply(probabilities.get(i) * (1 - probabilities.get(i)))
                            );
                            hessian.setSubMatrix(
                                    i * numberOfFeatures,
                                    (i + 1) * numberOfFeatures - 1,
                                    j * numberOfFeatures,
                                    (j + 1) * numberOfFeatures - 1,
                                    subMatrix
                            );
                        }
                    }
                }
            }
            return hessian;
        }
    }

    /**
     * Class implementing the likelihood function for the multi-class logistic regression model.
     */
    private class StochasticLikelihoodFunction extends AbstractStochasticFunctionUsingList<TrainingData.Entry> {
        public StochasticLikelihoodFunction() {
            // Using the method Arrays.asList so that the training data array is not duplicated. The newly created list
            // is backed by the existing array and any changes made to the list also "write through" to the array.
            this.data = trainingData;
        }

        /**
         * Computes the gradient of the likelihood function for the multi-class logistic regression model.
         *
         * @param weights The current weights vector.
         * @return The gradient vector of the logistic regression likelihood function.
         */
        @Override
        public Vector estimateGradient(Vector weights, int startIndex, int endIndex) {
            Matrix W = new Matrix(weights, numberOfFeatures);
            Matrix gradient = new Matrix(W.getRowDimension(), W.getColumnDimension());
            for (int i = startIndex; i < endIndex; i++) {
                Vector probabilities = W.transpose().multiply(data.get(i).features);
                Vector innerProductWithLastClass = Vectors.build(numberOfClasses, VectorType.DENSE);
                innerProductWithLastClass.set(0, numberOfClasses - 2, probabilities);
                probabilities = probabilities.sub(MatrixUtilities.computeLogSumExp(innerProductWithLastClass));
                probabilities = probabilities.map(Math::exp);
                if (data.get(i).label != numberOfClasses - 1) {
                    probabilities.set(data.get(i).label, probabilities.get(data.get(i).label) - 1);
                }
                for (int c = 0; c < numberOfClasses - 1; c++) {
                    gradient.setColumn(
                            c,
                            gradient.getColumn(c).add(data.get(i).features.mult(probabilities.get(c)))
                    );
                }
            }
            return Vectors.dense(gradient.getColumnPackedArrayCopy());
        }
    }
}
