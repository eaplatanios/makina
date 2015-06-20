package org.platanios.learn.logic.database;

import javax.persistence.*;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "GroundPredicates",
        catalog = "learn_logic")
public class DatabaseGroundFormula {
    private long id;
    private List<DatabaseGroundFormulaPredicate> groundPredicates;

    protected DatabaseGroundFormula() { }

    protected DatabaseGroundFormula(List<DatabaseGroundFormulaPredicate> groundPredicates) {
        this.groundPredicates = groundPredicates;
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

    @OneToMany(mappedBy = "groundFormula", cascade = CascadeType.ALL, orphanRemoval = true)
    public List<DatabaseGroundFormulaPredicate> getGroundPredicates() {
        return groundPredicates;
    }

    private void setGroundPredicates(List<DatabaseGroundFormulaPredicate> groundPredicates) {
        this.groundPredicates = groundPredicates;
    }
}
