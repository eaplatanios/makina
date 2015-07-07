package org.platanios.learn.logic.database;

import javax.persistence.*;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "Predicates",
        catalog = "learn_logic",
        uniqueConstraints = @UniqueConstraint(name = "uk_name", columnNames = "name"),
        indexes = {
                @Index(columnList = "id", name = "id_index"),
                @Index(columnList = "name", name = "name_index")
        })
public class DatabasePredicate {
    private long id;
    private String name;
    private List<DatabasePredicateArgumentType> argumentTypes;
    private List<DatabaseGroundPredicate> groundPredicates;

    private boolean closed = false;

    protected DatabasePredicate() { }

    @SuppressWarnings("unchecked")
    protected DatabasePredicate(List<DatabaseEntityType> argumentTypes) {
        List<DatabasePredicateArgumentType> argumentTypesList = new ArrayList<>();
        for (int argumentIndex = 0; argumentIndex < argumentTypes.size(); argumentIndex++)
            argumentTypesList.add(new DatabasePredicateArgumentType(this,
                                                                    argumentIndex,
                                                                    argumentTypes.get(argumentIndex)));
        setArgumentTypes(argumentTypesList);
    }

    protected DatabasePredicate(String name, List<DatabaseEntityType> argumentTypes) {
        this(argumentTypes);
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

    @OneToMany(
            fetch = FetchType.LAZY,
            mappedBy = "predicate",
            cascade = CascadeType.ALL,
            orphanRemoval = true
    )
    public List<DatabasePredicateArgumentType> getArgumentTypes() {
        return argumentTypes;
    }

    public void setArgumentTypes(List<DatabasePredicateArgumentType> argumentTypes) {
        this.argumentTypes = argumentTypes;
    }

    @OneToMany(
            fetch = FetchType.LAZY,
            mappedBy = "predicate",
            cascade = CascadeType.ALL,
            orphanRemoval = true
    )
    public List<DatabaseGroundPredicate> getGroundPredicates() {
        return groundPredicates;
    }

    public void setGroundPredicates(List<DatabaseGroundPredicate> groundPredicates) {
        this.groundPredicates = groundPredicates;
    }

    @Basic
    @Column(name = "closed")
    public boolean getClosed() {
        return closed;
    }

    public void setClosed(boolean closed) {
        this.closed = closed;
    }
}
