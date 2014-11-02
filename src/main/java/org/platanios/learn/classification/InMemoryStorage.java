package org.platanios.learn.classification;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorType;
import org.platanios.learn.math.matrix.Vectors;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.sql.Connection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class InMemoryStorage<T extends Vector, R extends Serializable> extends Storage<T, R> {
    private static final Logger logger = LogManager.getLogger("Classification / In-Memory Storage");

    private Map<R, T> featureMap;

    @SuppressWarnings("unchecked")
    protected Storage<T, R> loadFeatureMap(ObjectInputStream inputStream) {
        try {
            int numberOfKeys = inputStream.readInt();
            featureMap = new HashMap<>(numberOfKeys);
            for (int i = 0; i < numberOfKeys; i++) {
                R key = (R) inputStream.readObject();
                VectorType featuresVectorType = VectorType.values()[inputStream.readInt()];
                T featuresVector = (T) Vectors.build(inputStream, featuresVectorType);
                featureMap.put(key, featuresVector);
            }
            inputStream.close();
            logger.debug("Loaded the feature map from an input stream.");
            return this;
        } catch (Exception e) {
            logger.error("An exception was thrown while loading the feature map from an input stream.", e);
            return null;
        }
    }

    protected Storage<T, R> loadFeatureMap(Connection databaseConnection) {
        throw new NotImplementedException();
    }

    public void addFeatureMapping(R key, T features) {
        featureMap.put(key, features);
    }

    public void addFeatureMappings(Map<R, T> featureMappings) {
        featureMap.putAll(featureMappings);
    }

    public T getFeatureVector(R key) {
        return featureMap.get(key);
    }

    public Map<R, T> getFeatureVectors(List<R> keys) {
        Map<R, T> resultingFeatureVectors = new HashMap<>();
        for (R key : keys)
            resultingFeatureVectors.put(key, featureMap.get(key));
        return resultingFeatureVectors;
    }

    public Map<R, T> getFeatureMap() {
        return new HashMap<>(featureMap);
    }

    public boolean writeFeatureMapToStream(ObjectOutputStream outputStream) {
        try {
            outputStream.writeInt(featureMap.keySet().size());
            for (R key : featureMap.keySet()) {
                Vector features = featureMap.get(key);
                outputStream.writeObject(key);
                outputStream.writeInt(features.type().ordinal());
                features.writeToStream(outputStream);
            }
            outputStream.close();
            logger.debug("Wrote the feature map to an output stream.");
            return true;
        } catch (IOException e) {
            logger.error("An exception was thrown while writing the feature map to an output stream.", e);
            return false;
        }
    }
}
