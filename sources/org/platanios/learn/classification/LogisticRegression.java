package org.platanios.learn.classification;

import com.google.common.base.Preconditions;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.QuasiNewtonSolver;
import org.platanios.learn.optimization.function.AbstractFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LogisticRegression {
    private final QuasiNewtonSolver solver;

    private Vector[] trainingData;
    private Integer[] trainingDataLabels;
    private int trainingDataSize;
    private Vector weights;

    public LogisticRegression(Vector[] trainingData, Integer[] trainingDataLabels) {
        Preconditions.checkArgument(trainingData.length == trainingDataLabels.length);

        this.trainingDataLabels = trainingDataLabels;
        this.trainingData = trainingData;
        trainingDataSize = trainingData.length;
        weights = new Vector(trainingData[0].getDimension(), 0);
        solver = new QuasiNewtonSolver(new LikelihoodFunction(), weights.getArray());
        solver.setMethod(QuasiNewtonSolver.Method.BROYDEN_FLETCHER_GOLDFARB_SHANNO);
    }

    public void train() {
        weights = solver.solve();
    }

    public double predict(double[] point) {
        return 1 / (1 + Math.exp(-weights.innerProduct(new Vector(point))));
    }

    public double[] predict(double[][] points) {
        double[] predictions = new double[points.length];
        for (int i = 0; i < points.length; i++) {
            predictions[i] = predict(points[i]);
        }
        return predictions;
    }

    private class LikelihoodFunction extends AbstractFunction {
        public double computeValue(Vector weights) {
            double likelihood = 0;
            for (int n = 0; n < trainingDataSize; n++) {
                double tempDotProduct = weights.innerProduct(trainingData[n]);
                likelihood += trainingDataLabels[n] * tempDotProduct - Math.log(1 + Math.exp(tempDotProduct));
            }
            return -likelihood;
        }

        public Vector computeGradient(Vector weights) {
            Vector gradient = new Vector(weights.getDimension(), 0);
            for (int n = 0; n < trainingDataSize; n++) {
                double tempDotProduct = weights.innerProduct(trainingData[n]);
                double tempExp = Math.exp(-tempDotProduct);
                gradient = gradient.subtract(trainingData[n].multiply(trainingDataLabels[n] - 1 / (1 + tempExp)));
            }
            return gradient;
        }

        public Matrix computeHessian(Vector weights) {
            Matrix hessian = new Matrix(new double[weights.getDimension()][weights.getDimension()]);
            for (int n = 0; n < trainingDataSize; n++) {
                double tempDotProduct = weights.innerProduct(trainingData[n]);
                double mu = 1 / (1 + Math.exp(-tempDotProduct));
                hessian = hessian.add(trainingData[n].outerProduct(trainingData[n]).multiply(mu * (1 - mu)));
            }
            return hessian;
        }
    }
}
