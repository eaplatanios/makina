package org.platanios.learn.logic.database;

import javax.persistence.*;
import javax.validation.constraints.NotNull;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "GroundPredicateArguments",
        catalog = "learn_logic",
        uniqueConstraints = @UniqueConstraint(
                name = "uk_predicate_argument",
                columnNames = {"predicate_id", "argument_index"}
        ),
        indexes = @Index(columnList = "predicate_id", name = "predicate_id_index"))
public class DatabaseGroundPredicateArgument {
    private long id;
    private DatabaseGroundPredicate groundPredicate;
    private int argumentIndex;
    private long argumentValue;

    private DatabaseGroundPredicateArgument() { }

    protected DatabaseGroundPredicateArgument(DatabaseGroundPredicate groundPredicate,
                                              int argumentIndex,
                                              long argumentValue) {
        setGroundPredicate(groundPredicate);
        setArgumentIndex(argumentIndex);
        setArgumentValue(argumentValue);
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
    @JoinColumn(name = "predicate_id", nullable = false, foreignKey = @ForeignKey(name = "fk_ground_id"))
    @NotNull
    public DatabaseGroundPredicate getGroundPredicate() {
        return groundPredicate;
    }

    public void setGroundPredicate(DatabaseGroundPredicate groundPredicate) {
        this.groundPredicate = groundPredicate;
    }

    @Basic
    @Column(name = "argument_index", nullable = false)
    @NotNull
    public int getArgumentIndex() {
        return argumentIndex;
    }

    public void setArgumentIndex(int argumentIndex) {
        this.argumentIndex = argumentIndex;
    }

    @Basic
    @Column(name = "argument_value", nullable = false)
    @NotNull
    public long getArgumentValue() {
        return argumentValue;
    }

    public void setArgumentValue(long argumentValue) {
        this.argumentValue = argumentValue;
    }
}
