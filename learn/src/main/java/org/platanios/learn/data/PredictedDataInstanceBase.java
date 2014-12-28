package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
class PredictedDataInstanceBase<T extends Vector, S> extends LabeledDataInstanceBase<T, S> {
    protected double probability;

    public PredictedDataInstanceBase(String name, S label, Object source, double probability) {
        super(name, label, source);
        this.probability = probability;
    }

    public double probability() {
        return probability;
    }

    public void probability(double probability) {
        this.probability = probability;
    }

    @Override
    public PredictedDataInstanceWithFeatures<T, S> toDataInstance(T features) {
        return new PredictedDataInstance<>(name, features, label, source, probability);
    }
}
