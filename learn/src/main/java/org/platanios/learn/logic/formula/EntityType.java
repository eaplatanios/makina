package org.platanios.learn.logic.formula;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "EntityTypes",
        catalog = "learn_logic",
        indexes = {
                @Index(columnList = "id", name = "id_index"),
                @Index(columnList = "name", name = "name_index")
        },
        uniqueConstraints = @UniqueConstraint(name = "uk_name", columnNames = "name"))
public class EntityType {
    private long id;
    /** Note that this field is not used for checking equality between different argument type objects. */
    private String name;
    private Set<EntityTypeValue> allowedValues;

    public EntityType() { }

    public EntityType(String name) {
        setName(name);
    }

    public EntityType(String name, Set<Long> allowedValues) {
        setName(name);
        Set<EntityTypeValue> allowedEntityTypeValues = new HashSet<>();
        allowedValues.forEach(value -> allowedEntityTypeValues.add(new EntityTypeValue(this, value)));
        setAllowedValues(allowedEntityTypeValues);
    }

    public EntityType(long id, String name, Set<Long> allowedValues) {
        this(name, allowedValues);
        setId(id);
    }

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    @Column(name = "id")
    public long getId() {
        return id;
    }

    private void setId(long id) {
        this.id = id;
    }

    @Basic
    @Column(name = "name", nullable = false)
    @NotNull
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    @OneToMany(fetch = FetchType.EAGER, mappedBy = "entityType", cascade = CascadeType.ALL, orphanRemoval = true)
    public Set<EntityTypeValue> getAllowedValues() {
        return allowedValues;
    }

    public void setAllowedValues(Set<EntityTypeValue> allowedValues) {
        this.allowedValues = allowedValues;
    }

    @Transient
    public Set<Long> getPrimitiveAllowedValues() {
        return allowedValues.stream().map(EntityTypeValue::getValue).collect(Collectors.toSet());
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

        return true;
    }

    @Override
    public int hashCode() {
        return (int) (id ^ (id >>> 32));
    }

    @Override
    public String toString() {
        return name;
    }
}
