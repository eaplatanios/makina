package org.platanios.learn.logic.grounding;

import javax.persistence.*;
import javax.validation.constraints.NotNull;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "GroundFormulaPredicates",
        catalog = "learn_logic",
        indexes = { @Index(columnList = "ground_formula_id", name = "ground_formula_id_index") })
public class GroundFormulaPredicate {
    private long id;
    private GroundFormula groundFormula;
    private GroundPredicate groundPredicate;

    public GroundFormulaPredicate() { }

    public GroundFormulaPredicate(GroundFormula groundFormula,
                                  GroundPredicate groundPredicate) {
        setGroundFormula(groundFormula);
        setGroundPredicate(groundPredicate);
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
    @JoinColumn(name = "ground_formula_id", nullable = false, foreignKey = @ForeignKey(name = "fk_ground_formula"))
    @NotNull
    public GroundFormula getGroundFormula() {
        return groundFormula;
    }

    public void setGroundFormula(GroundFormula groundFormula) {
        this.groundFormula = groundFormula;
    }

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "ground_predicate_id", nullable = false, foreignKey = @ForeignKey(name = "fk_ground_predicate"))
    @NotNull
    public GroundPredicate getGroundPredicate() {
        return groundPredicate;
    }

    public void setGroundPredicate(GroundPredicate groundPredicate) {
        this.groundPredicate = groundPredicate;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        GroundFormulaPredicate that = (GroundFormulaPredicate) other;

        if (id != that.id)
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        return (int) (id ^ (id >>> 32));
    }
}
