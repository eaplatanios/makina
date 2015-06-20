package org.platanios.learn.logic.database;

import javax.persistence.*;
import javax.validation.constraints.NotNull;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "GroundFormulaPredicates",
        catalog = "learn_logic",
        indexes = { @Index(columnList = "ground_formula_id", name = "ground_formula_id_index") })
public class DatabaseGroundFormulaPredicate {
    private long id;
    private DatabaseGroundFormula groundFormula;
    private DatabaseGroundPredicate groundPredicate;

    private DatabaseGroundFormulaPredicate() { }

    public DatabaseGroundFormulaPredicate(DatabaseGroundFormula groundFormula,
                                          DatabaseGroundPredicate groundPredicate) {
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
    public DatabaseGroundFormula getGroundFormula() {
        return groundFormula;
    }

    public void setGroundFormula(DatabaseGroundFormula groundFormula) {
        this.groundFormula = groundFormula;
    }

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "ground_predicate_id", nullable = false, foreignKey = @ForeignKey(name = "fk_ground_predicate"))
    @NotNull
    public DatabaseGroundPredicate getGroundPredicate() {
        return groundPredicate;
    }

    public void setGroundPredicate(DatabaseGroundPredicate groundPredicate) {
        this.groundPredicate = groundPredicate;
    }
}
