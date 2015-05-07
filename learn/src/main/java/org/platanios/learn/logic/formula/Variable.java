package org.platanios.learn.logic.formula;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Variable<T> extends Term<T> {
    private String name;

    public Variable(long identifier, VariableType<T> type) {
        super(identifier, type);
    }

    public Variable<T> setName(String name) {
        this.name = name;
        return this;
    }

    public String getName() {
        return name;
    }

    @Override
    public String toString() {
        if (this.name != null)
            return name;
        else
            return Long.toString(identifier);
    }
}
