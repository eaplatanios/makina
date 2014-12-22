package org.platanios.learn.classification;

import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface Classifier<T extends Vector, S> extends Serializable {
    public ClassifierType type();
    public PredictedDataInstance<T, S> predict(PredictedDataInstance<T, S> dataInstance);
    public DataSet<PredictedDataInstance<T, S>> predict(DataSet<PredictedDataInstance<T, S>> dataInstances);
    public void writeObject(ObjectOutputStream outputStream) throws IOException;
    public void readObject(ObjectInputStream inputStream) throws IOException, ClassNotFoundException;
}
