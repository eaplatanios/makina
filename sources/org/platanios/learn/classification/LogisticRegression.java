package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Utilities;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.QuasiNewtonSolver;
import org.platanios.learn.optimization.function.AbstractFunction;

import java.util.Arrays;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LogisticRegression {
    private final QuasiNewtonSolver solver;
    private final Vector[] trainingData;
    private final Integer[] trainingDataLabels;
    private final int trainingDataSize;
    private final int numberOfFeatures;
    private final int numberOfClasses;

    private Matrix weights;

    public LogisticRegression(Vector[] trainingData, Integer[] trainingDataLabels) {
        this(trainingData, trainingDataLabels, false);
    }

    public LogisticRegression(Vector[] trainingData, Integer[] trainingDataLabels, boolean largeScale) {
        if (trainingData.length != trainingDataLabels.length) {
            throw new IllegalArgumentException("The number of provided data labels and data samples must match.");
        }

        this.trainingDataLabels = trainingDataLabels;
        this.trainingData = trainingData;
        trainingDataSize = trainingData.length;
        numberOfClasses = 1 + Arrays.asList(trainingDataLabels).stream().max(Integer::compare).get();
        numberOfFeatures = trainingData[0].getDimension();
        weights = new Matrix(trainingData[0].getDimension(), numberOfClasses - 1);
        if (!largeScale) {
            solver = new QuasiNewtonSolver.Builder(new LikelihoodFunction(), weights.getColumnPackedArrayCopy())
                    .method(QuasiNewtonSolver.Method.BROYDEN_FLETCHER_GOLDFARB_SHANNO)
                    .build();
        } else {
            solver = new QuasiNewtonSolver.Builder(new LikelihoodFunction(), weights.getColumnPackedArrayCopy())
                    .method(QuasiNewtonSolver.Method.LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO)
                    .build();
        }
    }

    public void train() {
        weights = new Matrix(trainingData[0].getDimension(), numberOfClasses);
        weights.setSubMatrix(
                0,
                weights.getRowDimension() - 1,
                0,
                numberOfClasses - 2,
                new Matrix(solver.solve().getArray(), numberOfFeatures)
        );
    }

    public double[] predict(double[] point) {
        Vector probabilities = weights.transpose().multiply(new Vector(point)).computeFunctionResult(Math::exp);
        return probabilities.divide(probabilities.computeSum()).getArrayCopy();
    }

    public double[][] predict(double[][] points) {
        double[][] probabilities = new double[points.length][];
        for (int i = 0; i < points.length; i++) {
            Vector predictions = weights.transpose().multiply(new Vector(points[i])).computeFunctionResult(Math::exp);
            probabilities[i] = predictions.divide(predictions.computeSum()).getArrayCopy();
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
        public double computeValue(Vector weights) {
            Matrix W = new Matrix(weights.getArray(), numberOfFeatures);
            double likelihood = 0;
            for (int n = 0; n < trainingDataSize; n++) {
                Vector innerProduct = W.transpose().multiply(trainingData[n]);
                if (trainingDataLabels[n] != numberOfClasses - 1) {
                    likelihood += innerProduct.getElement(trainingDataLabels[n]);
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
        public Vector computeGradient(Vector weights) {
            Matrix W = new Matrix(weights.getArray(), numberOfFeatures);
            Matrix gradient = new Matrix(W.getRowDimension(), W.getColumnDimension());
            for (int n = 0; n < trainingDataSize; n++) {
                Vector probabilities = W.transpose().multiply(trainingData[n]).computeFunctionResult(Math::exp);
                probabilities = probabilities.divide(probabilities.computeSum() + 1);
                if (trainingDataLabels[n] != numberOfClasses - 1) {
                    probabilities.getArray()[trainingDataLabels[n]] -= 1;
                }
                for (int c = 0; c < numberOfClasses - 1; c++) {
                    gradient.setColumn(
                            c,
                            gradient.getColumn(c).add(trainingData[n].multiply(probabilities.getElement(c)))
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
        public Matrix computeHessian(Vector weights) {
            Matrix W = new Matrix(weights.getArray(), numberOfFeatures);
            Matrix hessian = new Matrix(new double[weights.getDimension()][weights.getDimension()]);
            for (int n = 0; n < trainingDataSize; n++) {
                Vector probabilities = W.transpose().multiply(trainingData[n]).computeFunctionResult(Math::exp);
                probabilities = probabilities.divide(probabilities.computeSum() + 1);
                for (int i = 0; i < numberOfClasses - 1; i++) {
                    for (int j = 0; j <= i; j++) {
                        Matrix subMatrix = hessian.getSubMatrix(i * numberOfFeatures,
                                                                (i + 1) * numberOfFeatures - 1,
                                                                j * numberOfFeatures,
                                                                (j + 1) * numberOfFeatures - 1);
                        if (i != j) {
                            subMatrix = subMatrix.add(
                                    trainingData[n]
                                            .outerProduct(trainingData[n])
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
                                    trainingData[n]
                                            .outerProduct(trainingData[n])
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
}
