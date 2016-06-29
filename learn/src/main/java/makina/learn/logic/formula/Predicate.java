package makina.learn.logic.formula;

import com.google.common.base.Objects;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Predicate {
    private long id;
    private String name;
    private List<EntityType> argumentTypes;

    private boolean closed = false;

    public Predicate(long id, String name, List<EntityType> argumentTypes) {
        this.id = id;
        this.name = name;
        this.argumentTypes = argumentTypes;
    }

    public Predicate(long id, String name, List<EntityType> argumentTypes, boolean closed) {
        this(id, name, argumentTypes);
        this.closed = closed;
    }

    public long getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public List<EntityType> getArgumentTypes() {
        return argumentTypes;
    }

    public boolean getClosed() {
        return closed;
    }

    public final boolean isValidArgumentAssignment(List<? extends Term> arguments) {
        if (arguments.size() != argumentTypes.size())
            throw new IllegalArgumentException("The number of provided arguments is different than the number of " +
                                                       "arguments this predicate accepts.");
        boolean validArgumentTypes = true;
        for (int argumentIndex = 0; argumentIndex < arguments.size(); argumentIndex++)
            if (!arguments.get(argumentIndex).getType().equals(argumentTypes.get(argumentIndex))) {
                validArgumentTypes = false;
                break;
            }
        return validArgumentTypes;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        Predicate that = (Predicate) other;

        return Objects.equal(id, that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(id);
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(name).append("(");
        for (int argumentTypeIndex = 0; argumentTypeIndex < argumentTypes.size(); argumentTypeIndex++) {
            stringBuilder.append(argumentTypes.get(argumentTypeIndex).toString());
            if (argumentTypeIndex != argumentTypes.size() - 1)
                stringBuilder.append(", ");
        }
        return stringBuilder.append(")").toString();
    }

    public String toStringWithoutArgumentTypes() {
        return name;
    }
}
