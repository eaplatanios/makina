package org.platanios.learn.classification.active;

import com.google.common.base.Objects;

/**
 * @author Emmanouil Antonios Platanios
 */
public class EntropyScoringFunction extends ScoringFunction {
    @Override
    public Double computeInformationGainHeuristicValue(Learning learning, Learning.InstanceToLabel instanceToLabel) {
        return entropy(instanceToLabel.getProbability());
    }

    private double entropy(double probability) {
        if (probability > 0)
            return -probability * Math.log(probability) - (1 - probability) * Math.log(1 - probability);
        else
            return 0;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(this.getClass());
    }

    @Override
    public String toString() {
        return "ENTROPY";
    }
}
