package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Utilities;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.QuasiNewtonSolver;
import org.platanios.learn.optimization.Solver;
import org.platanios.learn.optimization.StochasticGradientDescentSolver;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.AbstractStochasticFunction;

import java.util.Arrays;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LogisticRegression {
    private final Solver solver;
    private final TrainingData.Entry[] trainingData;
    private final int trainingDataSize;
    private final int numberOfFeatures;
    private final int numberOfClasses;

    private Matrix weights;

    public static class Builder {
        private final TrainingData.Entry[] trainingData;
        private final int trainingDataSize;
        private final int numberOfFeatures;
        private final int numberOfClasses;

        private boolean stochastic = false;
        private boolean largeScale = false;
        private int maximumNumberOfIterations = 10000;
        private double pointChangeTolerance = 1e-10;
        private boolean checkForPointConvergence = true;

        // Parameters relevant to stochastic iterative optimization algorithms.
        private int maximumNumberOfIterationsWithNoPointChange = 1;
        private int batchSize = 100;
        private double tau = 10;
        private double kappa = 0.75;

        // Parameters relevant to quasi-Newton optimization algorithms.
        private int maximumNumberOfFunctionEvaluations = 1000000;
        private double objectiveChangeTolerance = 1e-10;
        private double gradientTolerance = 1e-6;
        private boolean checkForObjectiveConvergence = true;
        private boolean checkForGradientConvergence = true;
        private int m = 1;

        public Builder(TrainingData trainingData) {
            this.trainingData = trainingData.getData();
            Integer[] trainingDataLabels = new Integer[this.trainingData.length];
            for (int i = 0; i < this.trainingData.length; i++) {
                trainingDataLabels[i] = this.trainingData[i].label;
            }
            trainingDataSize = this.trainingData.length;
            numberOfClasses = 1 + Arrays.asList(trainingDataLabels).stream().max(Integer::compare).get();
            numberOfFeatures = this.trainingData[0].features.getDimension();
        }

        public Builder stochastic(boolean stochastic) {
            this.stochastic = stochastic;
            return this;
        }

        public Builder largeScale(boolean largeScale) {
            this.largeScale = largeScale;
            return this;
        }

        public Builder maximumNumberOfIterations(int maximumNumberOfIterations) {
            this.maximumNumberOfIterations = maximumNumberOfIterations;
            return this;
        }

        public Builder maximumNumberOfIterationsWithNoPointChange(int maximumNumberOfIterationsWithNoPointChange) {
            this.maximumNumberOfIterationsWithNoPointChange = maximumNumberOfIterationsWithNoPointChange;
            return this;
        }

        public Builder pointChangeTolerance(double pointChangeTolerance) {
            this.pointChangeTolerance = pointChangeTolerance;
            return this;
        }

        public Builder checkForPointConvergence(boolean checkForPointConvergence) {
            this.checkForPointConvergence = checkForPointConvergence;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder tau(double tau) {
            this.tau = tau;
            return this;
        }

        public Builder kappa(double kappa) {
            this.kappa = kappa;
            return this;
        }

        public Builder maximumNumberOfFunctionEvaluations(int maximumNumberOfFunctionEvaluations) {
            this.maximumNumberOfFunctionEvaluations = maximumNumberOfFunctionEvaluations;
            return this;
        }

        public Builder objectiveChangeTolerance(double objectiveChangeTolerance) {
            this.objectiveChangeTolerance = objectiveChangeTolerance;
            return this;
        }

        public Builder gradientTolerance(double gradientTolerance) {
            this.gradientTolerance = gradientTolerance;
            return this;
        }

        public Builder checkForObjectiveConvergence(boolean checkForObjectiveConvergence) {
            this.checkForObjectiveConvergence = checkForObjectiveConvergence;
            return this;
        }

        public Builder checkForGradientConvergence(boolean checkForGradientConvergence) {
            this.checkForGradientConvergence = checkForGradientConvergence;
            return this;
        }

        public Builder m(int m) {
            this.m = m;
            return this;
        }

        public LogisticRegression build() {
            return new LogisticRegression(this);
        }
    }

    public LogisticRegression(Builder builder) {
        trainingData = builder.trainingData;
        trainingDataSize = builder.trainingDataSize;
        numberOfFeatures = builder.numberOfFeatures;
        numberOfClasses = builder.numberOfClasses;
        weights = new Matrix(numberOfFeatures, numberOfClasses - 1);
        if (builder.stochastic) {
            solver = new StochasticGradientDescentSolver.Builder(new StochasticLikelihoodFunction(),
                                                                 weights.getColumnPackedArrayCopy())
                    .maximumNumberOfIterations(builder.maximumNumberOfIterations)
                    .maximumNumberOfIterationsWithNoPointChange(builder.maximumNumberOfIterationsWithNoPointChange)
                    .pointChangeTolerance(builder.pointChangeTolerance)
                    .checkForPointConvergence(builder.checkForPointConvergence)
                    .batchSize(builder.batchSize)
                    .tau(builder.tau)
                    .kappa(builder.kappa)
                    .build();
            return;
        }
        if (!builder.largeScale) {
            solver = new QuasiNewtonSolver.Builder(new LikelihoodFunction(), weights.getColumnPackedArrayCopy())
                    .method(QuasiNewtonSolver.Method.BROYDEN_FLETCHER_GOLDFARB_SHANNO)
                    .maximumNumberOfIterations(builder.maximumNumberOfIterations)
                    .maximumNumberOfFunctionEvaluations(builder.maximumNumberOfFunctionEvaluations)
                    .pointChangeTolerance(builder.pointChangeTolerance)
                    .objectiveChangeTolerance(builder.objectiveChangeTolerance)
                    .gradientTolerance(builder.gradientTolerance)
                    .checkForPointConvergence(builder.checkForPointConvergence)
                    .checkForObjectiveConvergence(builder.checkForObjectiveConvergence)
                    .checkForGradientConvergence(builder.checkForGradientConvergence)
                    .build();
        } else {
            solver = new QuasiNewtonSolver.Builder(new LikelihoodFunction(), weights.getColumnPackedArrayCopy())
                    .method(QuasiNewtonSolver.Method.LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO)
                    .m(builder.m)
                    .maximumNumberOfIterations(builder.maximumNumberOfIterations)
                    .maximumNumberOfFunctionEvaluations(builder.maximumNumberOfFunctionEvaluations)
                    .pointChangeTolerance(builder.pointChangeTolerance)
                    .objectiveChangeTolerance(builder.objectiveChangeTolerance)
                    .gradientTolerance(builder.gradientTolerance)
                    .checkForPointConvergence(builder.checkForPointConvergence)
                    .checkForObjectiveConvergence(builder.checkForObjectiveConvergence)
                    .checkForGradientConvergence(builder.checkForGradientConvergence)
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
                new Matrix(solver.solve().getArray(), numberOfFeatures)
        );
    }

    public double[] predict(double[] point) {
        Vector predictions = weights.transpose().multiply(new Vector(point));
        predictions = predictions.subtract(Utilities.computeLogSumExp(predictions));
        predictions = predictions.computeFunctionResult(Math::exp);
        return predictions.getArrayCopy();
    }

    public double[][] predict(double[][] points) {
        double[][] probabilities = new double[points.length][];
        for (int i = 0; i < points.length; i++) {
            Vector predictions = weights.transpose().multiply(new Vector(points[i]));
            predictions = predictions.subtract(Utilities.computeLogSumExp(predictions));
            predictions = predictions.computeFunctionResult(Math::exp);
            probabilities[i] = predictions.getArrayCopy();
        }
        return probabilities;
    }

    /**
     * Class implementing the likelihood function for the multi-class logistic regression model.
     */
    private class LikelihoodFunction extends AbstractFunction {
        /**
         * Computes the value of the likelihood function for the multi-class logistic regression model.
         *
         * @param   weights The current weights vector.
         * @return          The value of the logistic regression likelihood function.
         */
        @Override
        public double computeValue(Vector weights) {
            Matrix W = new Matrix(weights.getArray(), numberOfFeatures);
            double likelihood = 0;
            for (int n = 0; n < trainingDataSize; n++) {
                Vector innerProduct = W.transpose().multiply(trainingData[n].features);
                if (trainingData[n].label != numberOfClasses - 1) {
                    likelihood += innerProduct.getElement(trainingData[n].label);
                }
                Vector innerProductWithLastClass = new Vector(numberOfClasses);
                innerProductWithLastClass.setSubVector(0, numberOfClasses - 2, innerProduct);
                likelihood -= Utilities.computeLogSumExp(innerProductWithLastClass);
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
            Matrix W = new Matrix(weights.getArray(), numberOfFeatures);
            Matrix gradient = new Matrix(W.getRowDimension(), W.getColumnDimension());
            for (int n = 0; n < trainingDataSize; n++) {
                Vector probabilities = W.transpose().multiply(trainingData[n].features);
                Vector innerProductWithLastClass = new Vector(numberOfClasses);
                innerProductWithLastClass.setSubVector(0, numberOfClasses - 2, probabilities);
                probabilities = probabilities.subtract(Utilities.computeLogSumExp(innerProductWithLastClass));
                probabilities = probabilities.computeFunctionResult(Math::exp);
                if (trainingData[n].label != numberOfClasses - 1) {
                    probabilities.getArray()[trainingData[n].label] -= 1;
                }
                for (int c = 0; c < numberOfClasses - 1; c++) {
                    gradient.setColumn(
                            c,
                            gradient.getColumn(c).add(trainingData[n].features.multiply(probabilities.getElement(c)))
                    );
                }
            }
            return new Vector(gradient.getColumnPackedArrayCopy());
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
            Matrix W = new Matrix(weights.getArray(), numberOfFeatures);
            Matrix hessian = new Matrix(new double[weights.getDimension()][weights.getDimension()]);
            for (int n = 0; n < trainingDataSize; n++) {
                Vector probabilities = W.transpose().multiply(trainingData[n].features);
                Vector innerProductWithLastClass = new Vector(numberOfClasses);
                innerProductWithLastClass.setSubVector(0, numberOfClasses - 2, probabilities);
                probabilities = probabilities.subtract(Utilities.computeLogSumExp(innerProductWithLastClass));
                probabilities = probabilities.computeFunctionResult(Math::exp);
                for (int i = 0; i < numberOfClasses - 1; i++) {
                    for (int j = 0; j <= i; j++) {
                        Matrix subMatrix = hessian.getSubMatrix(i * numberOfFeatures,
                                                                (i + 1) * numberOfFeatures - 1,
                                                                j * numberOfFeatures,
                                                                (j + 1) * numberOfFeatures - 1);
                        if (i != j) {
                            subMatrix = subMatrix.add(
                                    trainingData[n].features
                                            .outerProduct(trainingData[n].features)
                                            .multiply(-probabilities.getElement(i) * probabilities.getElement(j))
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
                                    trainingData[n].features
                                            .outerProduct(trainingData[n].features)
                                            .multiply(probabilities.getElement(i) * (1 - probabilities.getElement(i)))
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
    private class StochasticLikelihoodFunction extends AbstractStochasticFunction<TrainingData.Entry> {
        public StochasticLikelihoodFunction() {
            this.data = Arrays.asList(trainingData);
        }

        /**
         * Computes the gradient of the likelihood function for the multi-class logistic regression model.
         *
         * @param weights The current weights vector.
         * @return The gradient vector of the logistic regression likelihood function.
         */
        @Override
        public Vector estimateGradient(Vector weights, List<TrainingData.Entry> dataBatch) {
            Matrix W = new Matrix(weights.getArray(), numberOfFeatures);
            Matrix gradient = new Matrix(W.getRowDimension(), W.getColumnDimension());
            for (TrainingData.Entry example : dataBatch) {
                Vector probabilities = W.transpose().multiply(example.features);
                Vector innerProductWithLastClass = new Vector(numberOfClasses);
                innerProductWithLastClass.setSubVector(0, numberOfClasses - 2, probabilities);
                probabilities = probabilities.subtract(Utilities.computeLogSumExp(innerProductWithLastClass));
                probabilities = probabilities.computeFunctionResult(Math::exp);
                if (example.label != numberOfClasses - 1) {
                    probabilities.getArray()[example.label] -= 1;
                }
                for (int c = 0; c < numberOfClasses - 1; c++) {
                    gradient.setColumn(
                            c,
                            gradient.getColumn(c).add(example.features.multiply(probabilities.getElement(c)))
                    );
                }
            }
            return new Vector(gradient.getColumnPackedArrayCopy());
        }
    }
}
