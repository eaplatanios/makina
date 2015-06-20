package org.platanios.learn.logic.database;

import javax.persistence.*;
import javax.validation.constraints.NotNull;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "PredicateArgumentTypes",
        catalog = "learn_logic",
        uniqueConstraints = @UniqueConstraint(
                name = "uk_predicate_argument_index",
                columnNames = {"predicate_id", "argument_index"}
        ),
        indexes = { @Index(columnList = "predicate_id", name = "predicate_index") })
public class DatabasePredicateArgumentType {
    private long id;
    private DatabasePredicate predicate;
    private int argumentIndex;
    private DatabaseEntityType argumentType;

    private DatabasePredicateArgumentType() { }

    protected DatabasePredicateArgumentType(DatabasePredicate predicate,
                                            int argumentIndex,
                                            DatabaseEntityType argumentType) {
        setPredicate(predicate);
        setArgumentIndex(argumentIndex);
        setArgumentType(argumentType);
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
    @JoinColumn(name = "predicate_id", nullable = false, foreignKey = @ForeignKey(name = "fk_predicate"))
    @NotNull
    public DatabasePredicate getPredicate() {
        return predicate;
    }

    public void setPredicate(DatabasePredicate predicate) {
        this.predicate = predicate;
    }

    @Basic
    @Column(name = "argument_index")
    public int getArgumentIndex() {
        return argumentIndex;
    }

    public void setArgumentIndex(int argumentIndex) {
        this.argumentIndex = argumentIndex;
    }

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "argument_type_id", nullable = false, foreignKey = @ForeignKey(name = "fk_argument_type"))
    @NotNull
    public DatabaseEntityType getArgumentType() {
        return argumentType;
    }

    public void setArgumentType(DatabaseEntityType argumentType) {
        this.argumentType = argumentType;
    }
}
