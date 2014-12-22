package org.platanios.learn.optimization.function;

import org.platanios.learn.data.DataInstanceBase;
import org.platanios.learn.data.DataSet;
import org.platanios.learn.math.matrix.Vector;

import java.util.Iterator;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractStochasticFunctionUsingDataSet<D extends DataInstanceBase> extends AbstractStochasticFunction {
    protected DataSet<D> dataSet;

    private Iterator<List<D>> dataIterator;
    private boolean oldSampleWithReplacement = sampleWithReplacement;
    private int oldBatchSize = 0;

    /** {@inheritDoc} */
    @Override
    public final Vector estimateGradient(Vector point, int batchSize) {
        if (dataIterator == null || oldSampleWithReplacement != sampleWithReplacement || oldBatchSize != batchSize) {
            dataIterator = dataSet.continuousRandomBatchIterator(batchSize, sampleWithReplacement);
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
}
