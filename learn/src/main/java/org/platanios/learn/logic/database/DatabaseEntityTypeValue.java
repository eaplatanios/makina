package org.platanios.learn.logic.database;

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
public class DatabaseEntityTypeValue {
    private long id;
    private DatabaseEntityType entityType;
    private long value;

    private DatabaseEntityTypeValue() { }

    protected DatabaseEntityTypeValue(DatabaseEntityType entityType, long value) {
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
    public DatabaseEntityType getEntityType() {
        return entityType;
    }

    public void setEntityType(DatabaseEntityType entityType) {
        this.entityType = entityType;
    }

    @Basic
    @Column(name = "value")
    public long getValue() {
        return value;
    }

    public void setValue(long value) {
        this.value = value;
    }
}
