package org.platanios.learn.classification;

import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.LabeledDataInstance;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;

import java.io.IOException;
import java.io.OutputStream;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface Classifier<T extends Vector, S> {
    public ClassifierType type();
    public PredictedDataInstance<T, S> predict(LabeledDataInstance<T, S> dataInstance);
    public DataSet<PredictedDataInstance<T, S>> predict(DataSet<? extends LabeledDataInstance<T, S>> dataInstances);
    public PredictedDataInstance<T, S> predictInPlace(PredictedDataInstance<T, S> dataInstance);
    public DataSet<PredictedDataInstance<T, S>> predictInPlace(DataSet<PredictedDataInstance<T, S>> dataInstances);
    public void write(OutputStream outputStream, boolean includeType) throws IOException;
}
