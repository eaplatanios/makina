package org.platanios.learn.logic.grounding;

import javax.persistence.*;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "GroundPredicates",
        catalog = "learn_logic")
public class GroundFormula {
    private long id;
    private List<GroundFormulaPredicate> groundPredicates;

    public GroundFormula() { }

    public GroundFormula(List<GroundFormulaPredicate> groundPredicates) {
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
    public List<GroundFormulaPredicate> getGroundPredicates() {
        return groundPredicates;
    }

    private void setGroundPredicates(List<GroundFormulaPredicate> groundPredicates) {
        this.groundPredicates = groundPredicates;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        GroundFormula that = (GroundFormula) other;

        if (id != that.id)
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        return (int) (id ^ (id >>> 32));
    }
}
