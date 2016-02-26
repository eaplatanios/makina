package org.platanios.learn.neural.network;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
class LayerVariable extends Variable {
    private final Layer layer;

    LayerVariable(int id, Layer layer) {
        super(id, layer.outputSize());
        this.layer = layer;
    }

    LayerVariable(int id, String name, Layer layer) {
        super(id, name, layer.outputSize());
        this.layer = layer;
    }

    @Override
    public Vector value(NetworkState state) {
        return layer.value(state);
    }
}
