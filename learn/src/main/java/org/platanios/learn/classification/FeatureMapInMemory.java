package org.platanios.learn.classification;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorType;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
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
    }

    @Override
    @SuppressWarnings("unchecked")
    public void loadFeatureMap(ObjectInputStream inputStream) {
        try {
            if (numberOfViews != inputStream.readInt()) {
                logger.error("This feature map was initialized for a number of views that is different than the " +
                                     "number of feature views stored in the input stream.");
                throw new RuntimeException("This feature map was initialized for a number of views that is different " +
                                                   "than the number of feature views stored in the input stream.");
            }
            int numberOfKeys = inputStream.readInt();
            featureMap = new HashMap<>(numberOfKeys);
            for (int i = 0; i < numberOfKeys; i++) {
                String name = (String) inputStream.readObject();
                List<T> features = new ArrayList<>(numberOfViews);
                for (int view = 0; view < numberOfViews; view++)
                    features.add((T) ((VectorType) inputStream.readObject()).buildVector(inputStream));
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
        if (!featureMap.containsKey(name))
            featureMap.put(name, new ArrayList<>(numberOfViews));
        featureMap.get(name).set(view, features);
    }

    @Override
    public void addSingleViewFeatureMappings(Map<String, T> featureMappings, int view) {
        for (String name : featureMappings.keySet())
            addSingleViewFeatureMappings(name, featureMappings.get(name), view);
    }

    @Override
    public void addFeatureMappings(String name, List<T> features) {
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
    public boolean writeFeatureMapToStream(ObjectOutputStream outputStream) {
        try {
            outputStream.writeInt(numberOfViews);
            outputStream.writeInt(featureMap.keySet().size());
            for (String name : featureMap.keySet()) {
                outputStream.writeObject(name);
                for (int view = 0; view < numberOfViews; view++) {
                    outputStream.writeObject(featureMap.get(name).get(view).type());
                    featureMap.get(name).get(view).write(outputStream);
                }
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
