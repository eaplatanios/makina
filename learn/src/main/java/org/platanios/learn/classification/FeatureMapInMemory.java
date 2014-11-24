package org.platanios.learn.classification;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.serialization.UnsafeSerializationUtilities;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.sql.Connection;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class FeatureMapInMemory<T extends Vector> extends FeatureMap<T> {
    private static final Logger logger = LogManager.getLogger("Classification / In-Memory Storage");

    private Map<String, List<T>> featureMap;

    protected FeatureMapInMemory(int numberOfViews) {
        super(numberOfViews);
        featureMap = new HashMap<>();
    }

    @Override
    @SuppressWarnings("unchecked")
    public void loadFeatureMap(InputStream inputStream) {
        try {
            numberOfViews = UnsafeSerializationUtilities.readInt(inputStream);
            int numberOfKeys = UnsafeSerializationUtilities.readInt(inputStream);
            featureMap = new HashMap<>(numberOfKeys);
            for (int i = 0; i < numberOfKeys; i++) {
                String name = UnsafeSerializationUtilities.readString(inputStream, 1024);
                List<T> features = new ArrayList<>(numberOfViews);
                for (int view = 0; view < numberOfViews; view++)
                    features.add((T) Vectors.build(inputStream));
                featureMap.put(name, features);
            }
            inputStream.close();
            logger.debug("Loaded the feature map from an input stream.");
        } catch (Exception e) {
            logger.error("An exception was thrown while loading the feature map from an input stream.", e);
        }
    }

    @Override
    public void loadFeatureMap(Connection databaseConnection) {
        throw new NotImplementedException();
    }

    @Override
    public void addSingleViewFeatureMappings(String name, T features, int view) {
        if (!featureMap.containsKey(name)) {
            featureMap.put(name, new ArrayList<>(numberOfViews));
            for (int viewCount = 0; viewCount < numberOfViews; viewCount++)
                featureMap.get(name).add(null);
        }
        featureMap.get(name).set(view, features);
    }

    @Override
    public void addSingleViewFeatureMappings(Map<String, T> featureMappings, int view) {
        for (String name : featureMappings.keySet())
            addSingleViewFeatureMappings(name, featureMappings.get(name), view);
    }

    @Override
    public void addFeatureMappings(String name, List<T> features) {
        // TODO: Throw exception if the features list is not of length numberOfViews.
        featureMap.put(name, features);
    }

    @Override
    public void addFeatureMappings(Map<String, List<T>> featureMappings) {
        featureMap.putAll(featureMappings);
    }

    @Override
    public T getSingleViewFeatureVector(String name, int view) {
        return featureMap.get(name).get(view);
    }

    @Override
    public Map<String, T> getSingleViewFeatureVectors(List<String> names, int view) {
        Map<String, T> resultingFeatureVectors = new HashMap<>();
        for (String name : names)
            resultingFeatureVectors.put(name, featureMap.get(name).get(view));
        return resultingFeatureVectors;
    }

    @Override
    public Map<String, T> getSingleViewFeatureMap(int view) {
        Map<String, T> resultingFeatureVectors = new HashMap<>();
        for (Map.Entry<String, List<T>> entry : featureMap.entrySet())
            resultingFeatureVectors.put(entry.getKey(), entry.getValue().get(view));
        return resultingFeatureVectors;
    }

    @Override
    public List<T> getFeatureVectors(String name) {
        return featureMap.get(name);
    }

    @Override
    public Map<String, List<T>> getFeatureVectors(List<String> names) {
        Map<String, List<T>> resultingFeatureVectors = new HashMap<>();
        for (String name : names)
            resultingFeatureVectors.put(name, featureMap.get(name));
        return resultingFeatureVectors;
    }

    @Override
    public Map<String, List<T>> getFeatureMap() {
        return new HashMap<>(featureMap);
    }

    @Override
    public boolean writeFeatureMapToStream(OutputStream outputStream) {
        try {
            UnsafeSerializationUtilities.writeInt(outputStream, numberOfViews);
            UnsafeSerializationUtilities.writeInt(outputStream, featureMap.keySet().size());
            for (String name : featureMap.keySet()) {
                UnsafeSerializationUtilities.writeString(outputStream, name);
                for (int view = 0; view < numberOfViews; view++)
                    featureMap.get(name).get(view).write(outputStream, true);
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
