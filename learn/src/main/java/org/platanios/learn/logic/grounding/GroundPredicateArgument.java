package org.platanios.learn.logic.grounding;

import org.platanios.learn.logic.formula.Predicate;

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
public class GroundPredicateArgument {
    private long id;
    private Predicate predicate;
    private GroundPredicate groundPredicate;
    private int argumentIndex;
    private long argumentValue;

    public GroundPredicateArgument() { }

    public GroundPredicateArgument(Predicate predicate,
                                   GroundPredicate groundPredicate,
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
    public Predicate getPredicate() {
        return predicate;
    }

    public void setPredicate(Predicate predicate) {
        this.predicate = predicate;
    }

    @ManyToOne(fetch = FetchType.EAGER, optional = false)
    @JoinColumn(
            name = "ground_predicate_id",
            nullable = false,
            foreignKey = @ForeignKey(name = "fk_ground_predicate_id")
    )
    @NotNull
    public GroundPredicate getGroundPredicate() {
        return groundPredicate;
    }

    public void setGroundPredicate(GroundPredicate groundPredicate) {
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

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        GroundPredicateArgument that = (GroundPredicateArgument) other;

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
        return Long.toString(argumentValue);
    }
}
