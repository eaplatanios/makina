package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataInstance<T extends Vector, S> {
    private final String name;
    private final T features;
    private final S label;
    private final Object source;

    public DataInstance(T features) {
        this.name = null;
        this.features = features;
        this.label = null;
        this.source = null;
    }

    public DataInstance(T features, S label) {
        this.name = null;
        this.features = features;
        this.label = label;
        this.source = null;
    }

    public DataInstance(T features, S label, String name) {
        this.name = name;
        this.features = features;
        this.label = label;
        this.source = null;
    }

    public DataInstance(T features, S label, Object source) {
        this.name = null;
        this.features = features;
        this.label = label;
        this.source = source;
    }

    public DataInstance(T features, S label, String name, Object source) {
        this.name = name;
        this.features = features;
        this.label = label;
        this.source = source;
    }

    public String getName() {
        return name;
    }

    public T getFeatures() {
        return features;
    }

    public S getLabel() {
        return label;
    }

    public Object getSource() {
        return source;
    }
}
