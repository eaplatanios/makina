package makina.learn.logic.formula;

import com.google.common.base.Objects;

import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public class EntityType {
    private long id;
    /** Note that this field is not used for checking equality between different argument type objects. */
    private String name;
    private Set<Long> allowedValues;

    public EntityType(long id, String name, Set<Long> allowedValues) {
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

    public Set<Long> getAllowedValues() {
        return allowedValues;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        EntityType that = (EntityType) other;

        return Objects.equal(id, that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(id);
    }

    @Override
    public String toString() {
        return name;
    }
}
