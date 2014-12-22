package org.platanios.learn.data;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.serialization.UnsafeSerializationUtilities;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Interface specifying methods for which all classes implementing a storage method for classification data (i.e.,
 * feature maps, data instances, etc.) should have implementations.
 *
 * @author Emmanouil Antonios Platanios
 */
public abstract class FeatureMap<T extends Vector> {
    static final Logger logger = LogManager.getLogger("Feature Map");

    protected int numberOfViews;

    FeatureMap() {

    }

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

    public int getNumberOfViews() {
        return numberOfViews;
    }

    public abstract List<String> getNames();
    public abstract void addFeatureMappings(String name, T features, int view);
    public abstract void addFeatureMappings(Map<String, T> featureMappings, int view);
    public abstract void addFeatureMappings(String name, List<T> features);
    public abstract void addFeatureMappings(Map<String, List<T>> featureMappings);
    public abstract T getFeatureVector(String name, int view);
    public abstract Map<String, T> getFeatureVectors(List<String> names, int view);
    public abstract Map<String, T> getFeatureMap(int view);
    public abstract List<T> getFeatureVectors(String name);
    public abstract Map<String, List<T>> getFeatureVectors(List<String> names);
    public abstract Map<String, List<T>> getFeatureMap();
    public abstract boolean write(OutputStream outputStream);

    @SuppressWarnings("unchecked")
    public static <T extends Vector> FeatureMap<T> read(InputStream inputStream, Type type) {
        try {
            int numberOfViews = UnsafeSerializationUtilities.readInt(inputStream);
            FeatureMap<T> featureMap = type.build(numberOfViews);
            int numberOfKeys = UnsafeSerializationUtilities.readInt(inputStream);
            for (int i = 0; i < numberOfKeys; i++) {
                String name = UnsafeSerializationUtilities.readString(inputStream, 1024);
                List<T> features = new ArrayList<>(numberOfViews);
                for (int view = 0; view < numberOfViews; view++)
                    features.add((T) Vectors.build(inputStream));
                featureMap.addFeatureMappings(name, features);
            }
            inputStream.close();
            logger.debug("Loaded the feature map from an input stream.");
            return featureMap;
        } catch (Exception e) {
            logger.error("An exception was thrown while loading the feature map from an input stream.", e);
            throw new RuntimeException(e);
        }
    }

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
