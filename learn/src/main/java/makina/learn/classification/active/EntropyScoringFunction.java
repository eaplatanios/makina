package makina.learn.classification.active;

import com.google.common.base.Objects;

/**
 * @author Emmanouil Antonios Platanios
 */
public class EntropyScoringFunction extends ScoringFunction {
    private final boolean propagateConstraints;

    public EntropyScoringFunction() {
        this(false);
    }

    public EntropyScoringFunction(boolean propagateConstraints) {
        this.propagateConstraints = propagateConstraints;
    }

    @Override
    public Double computeInformationGainHeuristicValue(Learning learning, Learning.InstanceToLabel instanceToLabel) {
        return entropy(instanceToLabel.getProbability());
    }

    private double entropy(double probability) {
        if (probability > 0 && probability < 1)
            return -probability * Math.log(probability) - (1 - probability) * Math.log(1 - probability);
        else
            return 0;
    }

    @Override
    public boolean propagateConstraints() {
        return propagateConstraints;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        EntropyScoringFunction that = (EntropyScoringFunction) other;

        return Objects.equal(propagateConstraints, that.propagateConstraints);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(this.getClass(), propagateConstraints);
    }

    @Override
    public String toString() {
        if (propagateConstraints)
            return "ENTROPY-CP";
        else
            return "ENTROPY";
    }
}
