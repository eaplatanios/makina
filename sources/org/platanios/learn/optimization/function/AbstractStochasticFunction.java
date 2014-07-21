package org.platanios.learn.optimization.function;

import org.platanios.learn.math.matrix.Vector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractStochasticFunction<T> {
    protected List<T> data;

    private int numberOfGradientEvaluations = 0;
    private boolean sampleWithReplacement = true;
    private int currentSampleIndex = 0;

    /**
     * Wrapper method for {@link #estimateGradient(org.platanios.learn.math.matrix.Vector, int)} which counts how many
     * times that method is called.
     *
     * @param   point   The point in which to estimate the derivatives.
     * @return          The values of the first derivatives of the objective function, estimated at the given point.
     */
    public final Vector getGradientEstimate(Vector point, int batchSize) {
        numberOfGradientEvaluations++;
        return estimateGradient(point, batchSize);
    }

    /**
     * Estimates the first derivatives of the objective function and the constraints at a particular point using a
     * subset of the available data. The size of that subset if given by {@code batchSize}.
     *
     * @param   point   The point in which to estimate the derivatives.
     * @return          The values of the first derivatives of the objective function, estimated at the given point.
     */
    public final Vector estimateGradient(Vector point, int batchSize) {
        if (batchSize < data.size()) {
            List<T> dataBatch;
            if (sampleWithReplacement) {
                Collections.shuffle(data);
                dataBatch = new ArrayList<>(data.subList(0, batchSize));
            } else {
                if (currentSampleIndex == 0 || currentSampleIndex >= data.size()) {
                    currentSampleIndex = 0;
                    Collections.shuffle(data);
                }
                dataBatch = new ArrayList<>(data.subList(currentSampleIndex, currentSampleIndex + batchSize));
                currentSampleIndex += batchSize;
            }
            return estimateGradient(point, dataBatch);
        } else {
            return estimateGradient(point, data);
        }
    }

    /**
     * Estimates the first derivatives of the objective function and the constraints at a particular point using the
     * given data.
     *
     * @param   point       The point in which to estimate the derivatives.
     * @param   dataBatch   The data to use to perform the gradient estimation.
     * @return              The values of the first derivatives of the objective function, estimated at the given point.
     */
    public abstract Vector estimateGradient(Vector point, List<T> dataBatch);

    public int getNumberOfGradientEvaluations() {
        return numberOfGradientEvaluations;
    }

    public final boolean getSampleWithReplacement() {
        return sampleWithReplacement;
    }

    public final void setSampleWithReplacement(boolean sampleWithReplacement) {
        this.sampleWithReplacement = sampleWithReplacement;
    }
}
