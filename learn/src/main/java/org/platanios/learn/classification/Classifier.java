package org.platanios.learn.classification;

import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.LabeledDataInstance;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.math.matrix.Vector;

import java.io.IOException;
import java.io.OutputStream;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface Classifier<T extends Vector, S> {
    ClassifierType type();
    PredictedDataInstance<T, S> predictInPlace(PredictedDataInstance<T, S> dataInstance);
    void write(OutputStream outputStream, boolean includeType) throws IOException;

    default PredictedDataInstance<T, S> predict(LabeledDataInstance<T, S> dataInstance) {
        return predictInPlace(new PredictedDataInstance<>(dataInstance.name(),
                                                          dataInstance.features(),
                                                          dataInstance.label(),
                                                          dataInstance.source(),
                                                          0));
    }

    default DataSet<PredictedDataInstance<T, S>> predict(
            DataSet<? extends LabeledDataInstance<T, S>> dataInstances
    ) {
        DataSet<PredictedDataInstance<T, S>> dataSet = dataInstances.newDataSet();
        for (LabeledDataInstance<T, S> dataInstance : dataInstances)
            dataSet.add(new PredictedDataInstance<>(dataInstance.name(),
                                                    dataInstance.features(),
                                                    dataInstance.label(),
                                                    dataInstance.source(),
                                                    0));
        return predictInPlace(dataSet);
    }

    /**
     * Predict the probabilities of the class labels being equal to 1 for a set of data instances. One probability computeValue
     * is provided for each data instance in the set.
     *
     * @param   dataSet The set of data instances for which the probabilities are computed in the form of an array.
     * @return          The probabilities of the class labels being equal to 1 for the given set of data instances.
     */
    default DataSet<PredictedDataInstance<T, S>> predictInPlace(
            DataSet<PredictedDataInstance<T, S>> dataSet
    ) {
        for (PredictedDataInstance<T, S> dataInstance : dataSet)
            predictInPlace(dataInstance);
        return dataSet;
    }
}
