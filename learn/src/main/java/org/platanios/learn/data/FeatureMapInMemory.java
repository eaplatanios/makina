package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.serialization.UnsafeSerializationUtilities;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class FeatureMapInMemory<T extends Vector> extends FeatureMap<T> {
    private Map<String, List<T>> featureMap;

    public FeatureMapInMemory(int numberOfViews) {
        super(numberOfViews);
        featureMap = new HashMap<>();
    }

    /** {@inheritDoc} */
    @Override
    public List<String> getNames() {
        return new ArrayList<>(featureMap.keySet());
    }

    /** {@inheritDoc} */
    @Override
    public void addFeatureMappings(String name, T features, int view) {
        if (!featureMap.containsKey(name)) {
            featureMap.put(name, new ArrayList<>(numberOfViews));
            for (int viewCount = 0; viewCount < numberOfViews; viewCount++)
                featureMap.get(name).add(null);
        }
        featureMap.get(name).set(view, features);
    }

    /** {@inheritDoc} */
    @Override
    public void addFeatureMappings(Map<String, T> featureMappings, int view) {
        for (String name : featureMappings.keySet())
            addFeatureMappings(name, featureMappings.get(name), view);
    }

    /** {@inheritDoc} */
    @Override
    public void addFeatureMappings(String name, List<T> features) {
        if (features.size() != numberOfViews) {
            logger.error("The size of the provided list must be equal to number of views.");
            throw new RuntimeException("The size of the provided list must be equal to number of views.");
        }
        featureMap.put(name, features);
    }

    /** {@inheritDoc} */
    @Override
    public void addFeatureMappings(Map<String, List<T>> featureMappings) {
        for (Map.Entry<String, List<T>> featureMapEntry : featureMappings.entrySet())
            if (featureMapEntry.getValue().size() != numberOfViews) {
                logger.error("All lists in the provided map must have size equal to number of views.");
                throw new RuntimeException("All lists in the provided map must have size equal to number of views.");
            }
        featureMap.putAll(featureMappings);
    }

    /** {@inheritDoc} */
    @Override
    public T getFeatureVector(String name, int view) {
        return featureMap.get(name).get(view);
    }

    /** {@inheritDoc} */
    @Override
    public Map<String, T> getFeatureVectors(List<String> names, int view) {
        Map<String, T> resultingFeatureVectors = new HashMap<>();
        for (String name : names)
            resultingFeatureVectors.put(name, featureMap.get(name).get(view));
        return resultingFeatureVectors;
    }

    /** {@inheritDoc} */
    @Override
    public Map<String, T> getFeatureMap(int view) {
        Map<String, T> resultingFeatureVectors = new HashMap<>();
        for (Map.Entry<String, List<T>> entry : featureMap.entrySet())
            resultingFeatureVectors.put(entry.getKey(), entry.getValue().get(view));
        return resultingFeatureVectors;
    }

    /** {@inheritDoc} */
    @Override
    public List<T> getFeatureVectors(String name) {
        return featureMap.get(name);
    }

    /** {@inheritDoc} */
    @Override
    public Map<String, List<T>> getFeatureVectors(List<String> names) {
        Map<String, List<T>> resultingFeatureVectors = new HashMap<>();
        for (String name : names)
            resultingFeatureVectors.put(name, featureMap.get(name));
        return resultingFeatureVectors;
    }

    /** {@inheritDoc} */
    @Override
    public Map<String, List<T>> getFeatureMap() {
        return new HashMap<>(featureMap);
    }

    /** {@inheritDoc} */
    @Override
    public boolean write(OutputStream outputStream) {
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
