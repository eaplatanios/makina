package makina.learn.data;

import com.google.common.base.Objects;
import makina.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class PredictedDataInstance<T extends Vector, S> extends LabeledDataInstance<T, S> {
    protected double probability;

    public PredictedDataInstance(String name, S label, Object source, double probability) {
        super(name, label, source);
        this.probability = probability;
    }

    public PredictedDataInstance(String name, T features, S label, Object source, double probability) {
        super(name, features, label, source);
        this.probability = probability;
    }

    public double probability() {
        return probability;
    }

    public void probability(double probability) {
        this.probability = probability;
    }

    @Override
    protected PredictedDataInstanceBase<T, S> toDataInstanceBase() {
        return new PredictedDataInstanceBase<>(name, label, source, probability);
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        PredictedDataInstance<?, ?> that = (PredictedDataInstance<?, ?>) other;

        return super.equals(that)
                && Objects.equal(probability, that.probability);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(super.hashCode(), probability);
    }
}
