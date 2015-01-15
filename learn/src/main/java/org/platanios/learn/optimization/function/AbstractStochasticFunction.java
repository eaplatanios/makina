package org.platanios.learn.optimization.function;

import java.util.Random;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractStochasticFunction {
    protected boolean sampleWithReplacement = true;
    protected int numberOfGradientEvaluations = 0;
    protected Random random = new Random();

    /**
     * Wrapper method for {@link #estimateGradient(org.platanios.learn.math.matrix.Vector, int)} which counts how many
     * times that method is called.
     *
     * @param   point       The point in which to estimate the derivatives.
     * @param   batchSize
     * @return              The values of the first derivatives of the objective function, estimated at the given point.
     */
    public final Vector getGradientEstimate(Vector point, int batchSize) {
        numberOfGradientEvaluations++;
        return estimateGradient(point, batchSize);
    }

    /**
     * Estimates the first derivatives of the objective function and the constraints at a particular point using a
     * subset of the available data. The size of that subset if given by {@code batchSize}.
     *
     * @param   point       The point in which to estimate the derivatives.
     * @param   batchSize
     * @return              The values of the first derivatives of the objective function, estimated at the given point.
     */
    public abstract Vector estimateGradient(Vector point, int batchSize);

    public final boolean getSampleWithReplacement() {
        return sampleWithReplacement;
    }

    public final void setSampleWithReplacement(boolean sampleWithReplacement) {
        this.sampleWithReplacement = sampleWithReplacement;
    }

    public int getNumberOfGradientEvaluations() {
        return numberOfGradientEvaluations;
    }
}
