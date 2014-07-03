package org.platanios.learn.classification;

import com.google.common.base.Preconditions;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.QuasiNewtonSolver;
import org.platanios.learn.optimization.function.AbstractFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LogisticRegression {
    private final QuasiNewtonSolver solver;

    private RealVector[] trainingData;
    private Integer[] trainingDataLabels;
    private int trainingDataSize;
    private RealVector weights;

    public LogisticRegression(RealVector[] trainingData, Integer[] trainingDataLabels) {
        Preconditions.checkArgument(trainingData.length == trainingDataLabels.length);

        this.trainingDataLabels = trainingDataLabels;
        this.trainingData = trainingData;
        trainingDataSize = trainingData.length;
        weights = new ArrayRealVector(trainingData[0].getDimension(), 0);
        solver = new QuasiNewtonSolver(new LikelihoodFunction(), weights.toArray());
        solver.setMethod(QuasiNewtonSolver.Method.BROYDEN_FLETCHER_GOLDFARB_SHANNO);
    }

    public void train() {
        weights = solver.solve();
    }

    public double predict(double[] point) {
        return 1 / (1 + Math.exp(-weights.dotProduct(new ArrayRealVector(point))));
    }

    public double[] predict(double[][] points) {
        double[] predictions = new double[points.length];
        for (int i = 0; i < points.length; i++) {
            predictions[i] = predict(points[i]);
        }
        return predictions;
    }

    private class LikelihoodFunction extends AbstractFunction {
        public double computeValue(RealVector weights) {
            double likelihood = 0;
            for (int n = 0; n < trainingDataSize; n++) {
                double tempDotProduct = weights.dotProduct(trainingData[n]);
                likelihood += trainingDataLabels[n] * tempDotProduct - Math.log(1 + Math.exp(tempDotProduct));
            }
            return -likelihood;
        }

        public RealVector computeGradient(RealVector weights) {
            RealVector gradient = new ArrayRealVector(weights.getDimension(), 0);
            for (int n = 0; n < trainingDataSize; n++) {
                double tempDotProduct = weights.dotProduct(trainingData[n]);
                double tempExp = Math.exp(-tempDotProduct);
                gradient = gradient.subtract(trainingData[n].mapMultiply(trainingDataLabels[n] - 1 / (1 + tempExp)));
            }
            return gradient;
        }

        public RealMatrix computeHessian(RealVector weights) {
            RealMatrix hessian = new Array2DRowRealMatrix(new double[weights.getDimension()][weights.getDimension()]);
            for (int n = 0; n < trainingDataSize; n++) {
                double tempDotProduct = weights.dotProduct(trainingData[n]);
                double mu = 1 / (1 + Math.exp(-tempDotProduct));
                hessian = hessian.add(trainingData[n].outerProduct(trainingData[n]).scalarMultiply(mu * (1 - mu)));
            }
            return hessian;
        }
    }
}
