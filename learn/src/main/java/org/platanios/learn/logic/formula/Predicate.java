package org.platanios.learn.logic.formula;

import org.platanios.learn.logic.grounding.GroundPredicate;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
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
public class Predicate {
    private long id;
    private String name;
    private List<PredicateArgumentType> argumentTypes;
    private List<GroundPredicate> groundPredicates;

    private boolean closed = false;

    public Predicate() { }

    public Predicate(String name, List<EntityType> argumentTypes) {
        setName(name);
        List<PredicateArgumentType> argumentTypesList = new ArrayList<>();
        for (int argumentIndex = 0; argumentIndex < argumentTypes.size(); argumentIndex++)
            argumentTypesList.add(new PredicateArgumentType(this, argumentIndex, argumentTypes.get(argumentIndex)));
        setArgumentTypes(argumentTypesList);
    }

    public Predicate(long id, String name, List<EntityType> argumentTypes) {
        this(name, argumentTypes);
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

    @OneToMany(
            fetch = FetchType.LAZY,
            mappedBy = "predicate",
            cascade = CascadeType.ALL,
            orphanRemoval = true
    )
    @NotNull
    public List<PredicateArgumentType> getArgumentTypes() {
        return argumentTypes;
    }

    public void setArgumentTypes(List<PredicateArgumentType> argumentTypes) {
        this.argumentTypes = argumentTypes;
    }

    @OneToMany(
            fetch = FetchType.LAZY,
            mappedBy = "predicate",
            cascade = CascadeType.ALL,
            orphanRemoval = true
    )
    public List<GroundPredicate> getGroundPredicates() {
        return groundPredicates;
    }

    public void setGroundPredicates(List<GroundPredicate> groundPredicates) {
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

    @Transient
    public final boolean isValidArgumentAssignment(List<? extends Term> arguments) {
        if (arguments.size() != argumentTypes.size())
            throw new IllegalArgumentException("The number of provided arguments is different than the number of " +
                                                       "arguments this predicate accepts.");
        boolean validArgumentTypes = true;
        for (int argumentIndex = 0; argumentIndex < arguments.size(); argumentIndex++)
            if (!arguments.get(argumentIndex).getType().equals(argumentTypes.get(argumentIndex).getArgumentType())) {
                validArgumentTypes = false;
                break;
            }
        return validArgumentTypes;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        Predicate that = (Predicate) other;

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
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(name).append("(");
        for (int argumentTypeIndex = 0; argumentTypeIndex < argumentTypes.size(); argumentTypeIndex++) {
            stringBuilder.append(argumentTypes.get(argumentTypeIndex).toString());
            if (argumentTypeIndex != argumentTypes.size() - 1)
                stringBuilder.append(", ");
        }
        return stringBuilder.append(")").toString();
    }

    public String toStringNoArgumentTypes() {
        return name;
    }
}
