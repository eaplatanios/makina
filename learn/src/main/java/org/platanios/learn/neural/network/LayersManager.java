package org.platanios.learn.neural.network;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author Emmanouil Antonios Platanios
 */
class LayersManager {
    private final AtomicInteger idCounter = new AtomicInteger(0);
    private final Map<Integer, Layer> layerIdsMap = new HashMap<>();

    LayersManager() { }

    private int id() {
        return idCounter.getAndIncrement();
    }

    int addLayer(Layer layer) {
        int id = id();
        layerIdsMap.put(id, layer);
        return id;
    }

    Layer get(int id) {
        return layerIdsMap.getOrDefault(id, null);
    }
}
