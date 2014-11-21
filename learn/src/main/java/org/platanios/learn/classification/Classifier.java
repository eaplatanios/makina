package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface Classifier<T extends Vector, S> extends Serializable {
    public ClassifierType type();
    public double predict(DataInstance<T, S> dataInstance);
    public double[] predict(List<DataInstance<T, S>> dataInstances);
    public List<DataInstance<T, S>> predictInPlace(List<DataInstance<T, S>> dataInstances);
    public void writeObject(ObjectOutputStream outputStream) throws IOException;
    public void readObject(ObjectInputStream inputStream) throws IOException, ClassNotFoundException;
}
