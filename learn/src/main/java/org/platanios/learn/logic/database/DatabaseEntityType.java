package org.platanios.learn.logic.database;

import javax.persistence.*;
import java.util.List;

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
public class DatabaseEntityType {
    private long id;
    /** Note that this field is not used for checking equality between different argument type objects. */
    private String name;
    private List<DatabaseEntityTypeValue> allowedValues;

    protected DatabaseEntityType() { }

    protected DatabaseEntityType(String name) {
        setName(name);
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
    @Column(name = "name")
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    @OneToMany(fetch = FetchType.LAZY, mappedBy = "entityType", cascade = CascadeType.ALL, orphanRemoval = true)
    public List<DatabaseEntityTypeValue> getAllowedValues() {
        return allowedValues;
    }

    public void setAllowedValues(List<DatabaseEntityTypeValue> allowedValues) {
        this.allowedValues = allowedValues;
    }
}
