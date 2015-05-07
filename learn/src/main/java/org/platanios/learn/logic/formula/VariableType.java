package org.platanios.learn.logic.formula;

/**
 * @author Emmanouil Antonios Platanios
 */
public class VariableType<T> {
    private final long identifier;
    private final Class<T> valueType;
    /** Note that this field is not used for checking equality between different argument type objects. */
    private String name;

    public VariableType(long identifier, Class<T> valueType) {
        this.identifier = identifier;
        this.valueType = valueType;
    }

    public long getIdentifier() {
        return identifier;
    }

    public VariableType<T> setName(String name) {
        this.name = name;
        return this;
    }

    public String getName() {
        return name;
    }

    public Class<T> getValueType() {
        return valueType;
    }

    @Override
    public boolean equals(Object object) {
        if (!(object instanceof VariableType))
            return false;
        if (object == this)
            return true;

        VariableType that = (VariableType) object;

        if (identifier != that.identifier)
            return false;
        if (valueType != that.valueType)
            return false;

        return true;
    }

    @Override
    public String toString() {
        if (name != null)
            return name;
        else
            return Long.toString(identifier);
    }
}
