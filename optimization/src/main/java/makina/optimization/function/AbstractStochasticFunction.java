package makina.optimization.function;

import makina.math.matrix.Vector;

import java.util.Random;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractStochasticFunction {
    protected boolean sampleWithReplacement = true;
    protected int numberOfGradientEvaluations = 0;
    protected Random random = new Random();

    /**
     * Wrapper method for {@link #estimateGradient(Vector, int)} which counts how many
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

    /**
     * Note that this function does not consider the Random object of this class when checking for equality.
     *
     * @param   other
     * @return
     */
    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        AbstractStochasticFunction that = (AbstractStochasticFunction) other;

        if (!super.equals(that))
            return false;
        if (sampleWithReplacement != that.sampleWithReplacement)
            return false;
        if (numberOfGradientEvaluations != that.numberOfGradientEvaluations)
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (sampleWithReplacement ? 1231 : 1237);
        result = 31 * result + numberOfGradientEvaluations;
        return result;
    }
}
