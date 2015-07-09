package org.platanios.learn.logic.formula;

import javax.persistence.*;
import javax.validation.constraints.NotNull;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "EntityTypeValues",
        catalog = "learn_logic",
        uniqueConstraints = @UniqueConstraint(
                name = "uk_entity_type_value",
                columnNames = {"entity_type_id", "value"}
        ),
        indexes = { @Index(columnList = "entity_type_id", name = "entity_type_id_index") })
public class EntityTypeValue {
    private long id;
    private EntityType entityType;
    private long value;

    public EntityTypeValue() { }

    public EntityTypeValue(EntityType entityType, long value) {
        setEntityType(entityType);
        setValue(value);
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

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "entity_type_id", nullable = false, foreignKey = @ForeignKey(name = "fk_entity_type"))
    @NotNull
    public EntityType getEntityType() {
        return entityType;
    }

    public void setEntityType(EntityType entityType) {
        this.entityType = entityType;
    }

    @Basic
    @Column(name = "value", nullable = false)
    @NotNull
    public long getValue() {
        return value;
    }

    public void setValue(long value) {
        this.value = value;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        EntityTypeValue that = (EntityTypeValue) other;

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
        return Long.toString(value);
    }
}
