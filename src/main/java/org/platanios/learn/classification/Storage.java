package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.sql.Connection;
import java.util.List;
import java.util.Map;

/**
 * Interface specifying methods for which all classes implementing a storage method for classification data (i.e.,
 * feature maps, data instances, etc.) should have implementations.
 *
 * @author Emmanouil Antonios Platanios
 */
public abstract class Storage<T extends Vector, R extends Serializable> {
    public static <T extends Vector, R extends Serializable> Storage<T, R> build(ObjectInputStream inputStream,
                                                                                 StorageType type) {
        return type.<T, R>build(inputStream);
    }

    public static <T extends Vector, R extends Serializable> Storage<T, R> build(Connection databaseConnection,
                                                                                 StorageType type) {
        return type.<T, R>build(databaseConnection);
    }

    protected abstract Storage<T, R> loadFeatureMap(ObjectInputStream inputStream);
    protected abstract Storage<T, R> loadFeatureMap(Connection databaseConnection);

    public abstract void addFeatureMapping(R key, T features);
    public abstract void addFeatureMappings(Map<R, T> featureMappings);
    public abstract T getFeatureVector(R key);
    public abstract Map<R, T> getFeatureVectors(List<R> keys);
    public abstract Map<R, T> getFeatureMap();
    public abstract boolean writeFeatureMapToStream(ObjectOutputStream outputStream);
}
