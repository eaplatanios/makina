package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class TrainingData {
    private Entry[] data;

    public TrainingData(Entry[] data) {
        this.data = data;
    }

    public Entry[] getData() {
        return data;
    }

    public static class Entry {
        public final Vector features;
        public final int label;

        public Entry(Vector features, int label) {
            this.features = features;
            this.label = label;
        }
    }
}
