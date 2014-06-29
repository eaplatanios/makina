package org.platanios.learn.classification;

import org.apache.commons.math3.linear.RealVector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class TrainingData {
    private RealVector[] data;
    private Integer[] labels;

    public TrainingData(RealVector[] data, Integer[] labels) {
        this.data = data;
        this.labels = labels;
    }

    public RealVector[] getData() {
        return data;
    }

    public Integer[] getLabels() {
        return labels;
    }
}
