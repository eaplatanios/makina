package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class TrainingData {
    private Vector[] data;
    private Integer[] labels;

    public TrainingData(Vector[] data, Integer[] labels) {
        this.data = data;
        this.labels = labels;
    }

    public Vector[] getData() {
        return data;
    }

    public Integer[] getLabels() {
        return labels;
    }
}
