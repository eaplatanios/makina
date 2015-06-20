package org.platanios.learn.logic.formula;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class EntityType {
    private final long id;
    /** Note that this field is not used for checking equality between different argument type objects. */
    private final String name;
    private final List<Long> allowedValues;

    public EntityType(long id, List<Long> allowedValues) {
        this.id = id;
        this.name = null;
        this.allowedValues = allowedValues;
    }

    public EntityType(long id, String name, List<Long> allowedValues) {
        this.id = id;
        this.name = name;
        this.allowedValues = allowedValues;
    }

    public long getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public List<Long> getAllowedValues() {
        return allowedValues;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        EntityType that = (EntityType) other;

        if (id != that.id)
            return false;
        if (name != null ? !name.equals(that.name) : that.name != null)
            return false;
        if (!allowedValues.equals(that.allowedValues))
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = (int) (id ^ (id >>> 32));
        result = 31 * result + (name != null ? name.hashCode() : 0);
        result = 31 * result + allowedValues.hashCode();
        return result;
    }

    @Override
    public String toString() {
        if (name != null)
            return name;
        else
            return Long.toString(id);
    }
}
