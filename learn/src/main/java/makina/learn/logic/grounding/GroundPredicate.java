package makina.learn.logic.grounding;

import com.google.common.base.Objects;
import makina.learn.logic.formula.Predicate;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GroundPredicate {
    private long id;
    private Predicate predicate;
    private List<Long> arguments;
    private Double value;

    public GroundPredicate(long id, Predicate predicate, List<Long> arguments) {
        this.id = id;
        this.predicate = predicate;
        this.arguments = arguments;
    }

    public GroundPredicate(long id, Predicate predicate, List<Long> arguments, Double value) {
        this(id, predicate, arguments);
        this.value = value;
    }

    public long getId() {
        return id;
    }

    public Predicate getPredicate() {
        return predicate;
    }

    public Double getValue() {
        return value;
    }

    public void setValue(Double value) {
        this.value = value;
    }

    public List<Long> getArguments() {
        return arguments;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        GroundPredicate that = (GroundPredicate) other;

        return Objects.equal(id, that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(id);
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(predicate.getName()).append("(");
        for (int argumentIndex = 0; argumentIndex < arguments.size(); argumentIndex++) {
            stringBuilder.append(arguments.get(argumentIndex).toString());
            if (argumentIndex != arguments.size() - 1)
                stringBuilder.append(", ");
        }
        return stringBuilder.append(")").toString();
    }
}
