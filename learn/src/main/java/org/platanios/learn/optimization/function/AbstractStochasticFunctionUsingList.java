package org.platanios.learn.optimization.function;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.statistics.StatisticsUtilities;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractStochasticFunctionUsingList<T> extends AbstractStochasticFunction {
    protected List<T> data;
    private int currentSampleIndex = 0;

    public void data(List<T> data) {
        this.data = data;
    }

    public List<T> data() {
        return data;
    }

    /** {@inheritDoc} */
    @Override
    public final Vector estimateGradient(Vector point, int batchSize) {
        if (batchSize < data.size()) {
            int startIndex;
            int endIndex;
            if (sampleWithReplacement) {
                StatisticsUtilities.shuffle(data, random);
                startIndex = 0;
                endIndex = batchSize;
            } else {
                if (currentSampleIndex == 0 || currentSampleIndex + batchSize >= data.size()) {
                    currentSampleIndex = 0;
                    StatisticsUtilities.shuffle(data, random);
                }
                startIndex = currentSampleIndex;
                endIndex = currentSampleIndex + batchSize;
                currentSampleIndex += batchSize;
            }
            return estimateGradient(point, startIndex, endIndex);
        } else {
            return estimateGradient(point, 0, data.size());
        }
    }

    /**
     * Estimates the first derivatives of the objective function and the constraints at a particular point using the
     * given data.
     *
     * @param   point       The point in which to estimate the derivatives.
     * @param   startIndex
     * @param   endIndex
     * @return              The values of the first derivatives of the objective function, estimated at the given point.
     */
    public abstract Vector estimateGradient(Vector point, int startIndex, int endIndex);

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        AbstractStochasticFunctionUsingList that = (AbstractStochasticFunctionUsingList) other;

        if (!super.equals(that))
            return false;
        if (!data.equals(that.data))
            return false;
        if (currentSampleIndex != that.currentSampleIndex)
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + data.hashCode();
        result = 31 * result + currentSampleIndex;
        return result;
    }
}
