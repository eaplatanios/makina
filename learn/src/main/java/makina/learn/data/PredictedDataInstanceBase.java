package makina.learn.data;

import com.google.common.base.Objects;
import makina.math.matrix.Vector;

import java.util.List;

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
    public PredictedDataInstance<T, S> toDataInstance(T features) {
        return new PredictedDataInstance<>(name, features, label, source, probability);
    }

    @Override
    public MultiViewDataInstance<T> toMultiViewDataInstance(List<T> features) {
        return new MultiViewPredictedDataInstance<>(name, features, label, source, probability);
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        PredictedDataInstanceBase<?, ?> that = (PredictedDataInstanceBase<?, ?>) other;

        return super.equals(that)
                && Objects.equal(probability, that.probability);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(super.hashCode(), probability);
    }
}
