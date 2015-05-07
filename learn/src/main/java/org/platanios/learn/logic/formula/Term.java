package org.platanios.learn.logic.formula;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class Term<T> {
    long identifier;
    VariableType<T> type;

    public Term(long identifier, VariableType<T> type) {
        this.identifier = identifier;
        this.type = type;
    }

    public long getIdentifier() {
        return identifier;
    }

    public VariableType<T> getType() {
        return type;
    }

    @Override
    public abstract String toString();
}
