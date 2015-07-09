package org.platanios.learn.logic.grounding;

import org.platanios.learn.logic.formula.Predicate;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "GroundPredicates",
        catalog = "learn_logic")
public class GroundPredicate {
    private long id;
    private Predicate predicate;
    private Double value;
    private List<GroundPredicateArgument> groundPredicateArguments;

    public GroundPredicate() { }

    public GroundPredicate(Predicate predicate, Double value) {
        setPredicate(predicate);
        setValue(value);
    }

    public GroundPredicate(Predicate predicate,
                           List<Long> predicateArgumentsAssignment) {
        setPredicate(predicate);
        List<GroundPredicateArgument> groundPredicateArguments = new ArrayList<>();
        for (int argumentIndex = 0; argumentIndex < predicateArgumentsAssignment.size(); argumentIndex++)
            groundPredicateArguments.add(new GroundPredicateArgument(
                    predicate,
                    this,
                    argumentIndex,
                    predicateArgumentsAssignment.get(argumentIndex)
            ));
        setGroundPredicateArguments(groundPredicateArguments);
    }

    public GroundPredicate(long id,
                           Predicate predicate,
                           List<Long> predicateArgumentsAssignment) {
        this(predicate, predicateArgumentsAssignment);
        setId(id);
    }

    public GroundPredicate(Predicate predicate,
                           List<Long> predicateArgumentsAssignment,
                           Double value) {
        this(predicate, predicateArgumentsAssignment);
        setValue(value);
    }

    public GroundPredicate(long id,
                           Predicate predicate,
                           List<Long> predicateArgumentsAssignment,
                           Double value) {
        this(id, predicate, predicateArgumentsAssignment);
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

    @ManyToOne(fetch = FetchType.EAGER, optional = false)
    @JoinColumn(name = "predicate_id", nullable = false, foreignKey = @ForeignKey(name = "fk_predicate"))
    @NotNull
    public Predicate getPredicate() {
        return predicate;
    }

    public void setPredicate(Predicate predicate) {
        this.predicate = predicate;
    }

    @Basic
    @Column(name = "value")
    public Double getValue() {
        return value;
    }

    public void setValue(Double value) {
        this.value = value;
    }

    @OneToMany(
            fetch = FetchType.EAGER,
            mappedBy = "groundPredicate",
            cascade = CascadeType.ALL,
            orphanRemoval = true
    )
    @NotNull
    public List<GroundPredicateArgument> getGroundPredicateArguments() {
        return groundPredicateArguments;
    }

    public void setGroundPredicateArguments(List<GroundPredicateArgument> groundPredicateArguments) {
        this.groundPredicateArguments = groundPredicateArguments;
    }

    @Transient
    public List<Long> getPredicateArgumentsAssignment() {
        return groundPredicateArguments
                .stream()
                .sorted((a1, a2) -> Integer.compare(a1.getArgumentIndex(), a2.getArgumentIndex()))
                .map(GroundPredicateArgument::getArgumentValue)
                .collect(Collectors.toList());
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        GroundPredicate that = (GroundPredicate) other;

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
        stringBuilder.append(predicate.getName()).append("(");
        for (int argumentIndex = 0; argumentIndex < groundPredicateArguments.size(); argumentIndex++) {
            stringBuilder.append(groundPredicateArguments.get(argumentIndex).toString());
            if (argumentIndex != groundPredicateArguments.size() - 1)
                stringBuilder.append(", ");
        }
        return stringBuilder.append(")").toString();
    }
}
