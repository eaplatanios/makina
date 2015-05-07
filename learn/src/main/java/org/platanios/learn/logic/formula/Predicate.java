package org.platanios.learn.logic.formula;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Predicate<T> {
    private final long identifier;
    private final List<VariableType<T>> argumentTypes;
    private String name;

    public Predicate(long identifier, List<VariableType<T>> argumentTypes) {
        this.identifier = identifier;
        this.argumentTypes = argumentTypes;
    }

    public Predicate<T> setName(String name) {
        this.name = name;
        return this;
    }

    public long getIdentifier() {
        return identifier;
    }

    public String getName() {
        return name;
    }

    public final boolean isValidArgumentAssignment(List<? extends Term<T>> arguments) {
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
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        if (name != null)
            stringBuilder.append(name).append("(");
        else
            stringBuilder.append(identifier).append("(");
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
            return Long.toString(identifier);
    }
}
