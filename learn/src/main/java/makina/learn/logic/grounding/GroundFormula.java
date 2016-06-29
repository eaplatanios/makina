package makina.learn.logic.grounding;

import com.google.common.base.Objects;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GroundFormula {
    private List<GroundPredicate> groundPredicates;

    public GroundFormula(List<GroundPredicate> groundPredicates) {
        this.groundPredicates = groundPredicates;
    }

    public List<GroundPredicate> getGroundPredicates() {
        return groundPredicates;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        GroundFormula that = (GroundFormula) other;

        return Objects.equal(groundPredicates, that.groundPredicates);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(groundPredicates);
    }
}
