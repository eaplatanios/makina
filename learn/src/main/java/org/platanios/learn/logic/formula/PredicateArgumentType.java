package org.platanios.learn.logic.formula;

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
public class PredicateArgumentType {
    private long id;
    private Predicate predicate;
    private int argumentIndex;
    private EntityType argumentType;

    public PredicateArgumentType() { }

    public PredicateArgumentType(Predicate predicate,
                                 int argumentIndex,
                                 EntityType argumentType) {
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
    public Predicate getPredicate() {
        return predicate;
    }

    public void setPredicate(Predicate predicate) {
        this.predicate = predicate;
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

    @ManyToOne(fetch = FetchType.EAGER, optional = false)
    @JoinColumn(name = "argument_type_id", nullable = false, foreignKey = @ForeignKey(name = "fk_argument_type"))
    @NotNull
    public EntityType getArgumentType() {
        return argumentType;
    }

    public void setArgumentType(EntityType argumentType) {
        this.argumentType = argumentType;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        PredicateArgumentType that = (PredicateArgumentType) other;

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
        return argumentType.toString();
    }
}
