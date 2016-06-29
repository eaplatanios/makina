package makina.learn.classification.active;

import com.google.common.base.Objects;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ProbabilityScoringFunction extends ScoringFunction {
    private final boolean propagateConstraints;

    public ProbabilityScoringFunction() {
        this(false);
    }

    public ProbabilityScoringFunction(boolean propagateConstraints) {
        this.propagateConstraints = propagateConstraints;
    }

    @Override
    public Double computeInformationGainHeuristicValue(Learning learning, Learning.InstanceToLabel instanceToLabel) {
        return instanceToLabel.getProbability();
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

        ProbabilityScoringFunction that = (ProbabilityScoringFunction) other;

        return Objects.equal(propagateConstraints, that.propagateConstraints);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(this.getClass(), propagateConstraints);
    }

    @Override
    public String toString() {
        if (!propagateConstraints)
            return "PROBABILITY";
        else
            return "PROBABILITY-CP";
    }
}
