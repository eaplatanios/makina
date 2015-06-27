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
                name = "uk_ground_predicate_argument_index",
                columnNames = {"ground_predicate_id", "argument_index"}
        ),
        indexes = {
                @Index(columnList = "predicate_id", name = "predicate_id_index"),
                @Index(columnList = "ground_predicate_id", name = "ground_predicate_id_index"),
                @Index(columnList = "argument_index", name = "argument_index_index"),
                @Index(columnList = "argument_value", name = "argument_value_index")
        })
public class DatabaseGroundPredicateArgument {
    private long id;
    private DatabasePredicate predicate;
    private DatabaseGroundPredicate groundPredicate;
    private int argumentIndex;
    private long argumentValue;

    protected DatabaseGroundPredicateArgument() { }

    protected DatabaseGroundPredicateArgument(DatabasePredicate predicate,
                                              DatabaseGroundPredicate groundPredicate,
                                              int argumentIndex,
                                              long argumentValue) {
        setPredicate(predicate);
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
    @JoinColumn(name = "predicate_id", nullable = false, foreignKey = @ForeignKey(name = "fk_predicate_id"))
    @NotNull
    public DatabasePredicate getPredicate() {
        return predicate;
    }

    public void setPredicate(DatabasePredicate predicate) {
        this.predicate = predicate;
    }

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "ground_predicate_id", nullable = false, foreignKey = @ForeignKey(name = "fk_ground_predicate_id"))
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
