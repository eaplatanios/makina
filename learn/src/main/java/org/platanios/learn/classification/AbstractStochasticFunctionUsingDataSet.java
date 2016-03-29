package org.platanios.learn.classification;

import org.platanios.learn.data.DataInstance;
import org.platanios.learn.data.DataSet;
import org.platanios.math.matrix.Vector;
import org.platanios.optimization.function.AbstractStochasticFunction;

import java.util.Iterator;
import java.util.List;

/**
 * TODO: Move into the learn module and remove the dependency of the optimization module to the learn module.
 *
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractStochasticFunctionUsingDataSet<D extends DataInstance>
        extends AbstractStochasticFunction {
    protected DataSet<D> dataSet;

    private Iterator<List<D>> dataIterator;
    private boolean oldSampleWithReplacement = sampleWithReplacement;
    private int oldBatchSize = 0;

    /** {@inheritDoc} */
    @Override
    public final Vector estimateGradient(Vector point, int batchSize) {
        if (dataIterator == null || oldSampleWithReplacement != sampleWithReplacement || oldBatchSize != batchSize) {
            dataIterator = dataSet.continuousRandomBatchIterator(batchSize, sampleWithReplacement, random);
            oldSampleWithReplacement = sampleWithReplacement;
            oldBatchSize = batchSize;
        }
        return estimateGradient(point, dataIterator.next());
    }

    /**
     * Estimates the first derivatives of the objective function and the constraints at a particular point using the
     * given data.
     *
     * @param   point       The point in which to estimate the derivatives.
     * @param   dataBatch
     * @return              The values of the first derivatives of the objective function, estimated at the given point.
     */
    public abstract Vector estimateGradient(Vector point, List<D> dataBatch);

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        AbstractStochasticFunctionUsingDataSet that = (AbstractStochasticFunctionUsingDataSet) other;

        if (!super.equals(that))
            return false;
        if (!dataSet.equals(that.dataSet))
            return false;
        if (!dataIterator.equals(that.dataIterator))
            return false;
        if (oldSampleWithReplacement != that.oldSampleWithReplacement)
            return false;
        if (oldBatchSize != that.oldBatchSize)
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + dataSet.hashCode();
        result = 31 * result + dataIterator.hashCode();
        result = 31 * result + (oldSampleWithReplacement ? 1231 : 1237);
        result = 31 * result + oldBatchSize;
        return result;
    }
}
