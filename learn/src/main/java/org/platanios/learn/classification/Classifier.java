package org.platanios.learn.classification;

import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;

import java.io.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface Classifier<T extends Vector, S> {
    public ClassifierType type();
    public PredictedDataInstance<T, S> predict(PredictedDataInstance<T, S> dataInstance);
    public DataSet<PredictedDataInstance<T, S>> predict(DataSet<PredictedDataInstance<T, S>> dataInstances);
    public void write(OutputStream outputStream) throws IOException;
}
