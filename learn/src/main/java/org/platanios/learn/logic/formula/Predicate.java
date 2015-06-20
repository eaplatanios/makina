package org.platanios.learn.logic.formula;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Predicate {
    private final long id;
    private final String name;
    private final List<EntityType> argumentTypes;

    public Predicate(long id, List<EntityType> argumentTypes) {
        this.id = id;
        this.name = null;
        this.argumentTypes = argumentTypes;
    }

    public Predicate(long id, String name, List<EntityType> argumentTypes) {
        this.id = id;
        this.name = name;
        this.argumentTypes = argumentTypes;
    }

    public long getId() {
        return id;
    }

    public String getName() {
        return name;
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

        if (id != that.id)
            return false;
        if (name != null ? !name.equals(that.name) : that.name != null)
            return false;
        if (!argumentTypes.equals(that.argumentTypes))
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = (int) (id ^ (id >>> 32));
        result = 31 * result + (name != null ? name.hashCode() : 0);
        result = 31 * result + argumentTypes.hashCode();
        return result;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        if (name != null)
            stringBuilder.append(name).append("(");
        else
            stringBuilder.append(id).append("(");
        for (int argumentTypeIndex = 0; argumentTypeIndex < argumentTypes.size(); argumentTypeIndex++) {
            stringBuilder.append(argumentTypes.get(argumentTypeIndex).toString());
            if (argumentTypeIndex != argumentTypes.size() - 1)
                stringBuilder.append(", ");
        }
        return stringBuilder.append(")").toString();
    }

    public String toStringNoArgumentTypes() {
        if (name != null)
            return name;
        else
            return Long.toString(id);
    }
}
