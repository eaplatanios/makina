package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.InputStream;
import java.io.OutputStream;
import java.sql.Connection;
import java.util.List;
import java.util.Map;

/**
 * Interface specifying methods for which all classes implementing a storage method for classification data (i.e.,
 * feature maps, data instances, etc.) should have implementations.
 *
 * @author Emmanouil Antonios Platanios
 */
public abstract class FeatureMap<T extends Vector> {
    protected int numberOfViews;

    protected FeatureMap(int numberOfViews) {
        this.numberOfViews = numberOfViews;
    }

    public static <T extends Vector> FeatureMap<T> build(int numberOfViews, Type type) {
        return type.<T>build(numberOfViews);
    }

    public static <T extends Vector> FeatureMap<T> buildMariaDB(int numberOfViews,
                                                                String host,
                                                                String username,
                                                                String password) {
        return new FeatureMapMariaDB<>(numberOfViews, host, username, password);
    }

    public abstract void loadFeatureMap(InputStream inputStream);
    public abstract void loadFeatureMap(Connection databaseConnection);
    public abstract void addSingleViewFeatureMappings(String name, T features, int view);
    public abstract void addSingleViewFeatureMappings(Map<String, T> featureMappings, int view);
    public abstract void addFeatureMappings(String name, List<T> features);
    public abstract void addFeatureMappings(Map<String, List<T>> featureMappings);
    public abstract T getSingleViewFeatureVector(String name, int view);
    public abstract Map<String, T> getSingleViewFeatureVectors(List<String> names, int view);
    public abstract Map<String, T> getSingleViewFeatureMap(int view);
    public abstract List<T> getFeatureVectors(String name);
    public abstract Map<String, List<T>> getFeatureVectors(List<String> names);
    public abstract Map<String, List<T>> getFeatureMap();
    public abstract boolean writeFeatureMapToStream(OutputStream outputStream);

    public enum Type {
        IN_MEMORY {
            @Override
            protected <T extends Vector> FeatureMap<T> build(int numberOfViews) {
                return new FeatureMapInMemory<>(numberOfViews);
            }
        },
        MARIA_DB {
            @Override
            protected <T extends Vector> FeatureMap<T> build(int numberOfViews) {
                throw new NotImplementedException();
            }
        };

        protected abstract <T extends Vector> FeatureMap<T> build(int numberOfViews);
    }
}
