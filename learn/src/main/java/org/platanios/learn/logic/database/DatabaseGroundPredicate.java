package org.platanios.learn.logic.database;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "GroundPredicates",
        catalog = "learn_logic")
public class DatabaseGroundPredicate {
    private long id;
    private DatabasePredicate predicate;
    private String value;
    private Class valueClass;
    private List<DatabaseGroundPredicateArgument> groundPredicateArguments;

    private DatabaseGroundPredicate() { }

    protected DatabaseGroundPredicate(DatabasePredicate predicate, String value, Class valueClass) {
        setPredicate(predicate);
        setValue(value);
        setValueClass(valueClass);
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
    public DatabasePredicate getPredicate() {
        return predicate;
    }

    public void setPredicate(DatabasePredicate predicate) {
        this.predicate = predicate;
    }

//    @Any(metaColumn = @Column(name = "value_type"))
//    @Cascade(CascadeType.ALL)
//    @AnyMetaDef(
//            idType = "long",
//            metaType = "string",
//            metaValues = {
//                    @MetaValue(value = "Boolean", targetEntity = Boolean.class),
//                    @MetaValue(value = "Integer", targetEntity = Integer.class),
//                    @MetaValue(value = "Long", targetEntity = Long.class),
//                    @MetaValue(value = "Float", targetEntity = Float.class),
//                    @MetaValue(value = "Double", targetEntity = Double.class)
//            })
//    @JoinColumn(name = "value_id")
    @Basic
    @Column(name = "value")
    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }

    @Basic
    @Column(name = "value_class", nullable = false)
    @NotNull
    public Class getValueClass() {
        return valueClass;
    }

    public void setValueClass(Class valueClass) {
        this.valueClass = valueClass;
    }

    @OneToMany(
            fetch = FetchType.LAZY,
            mappedBy = "groundPredicate",
            cascade = javax.persistence.CascadeType.ALL,
            orphanRemoval = true
    )
    public List<DatabaseGroundPredicateArgument> getGroundPredicateArguments() {
        return groundPredicateArguments;
    }

    public void setGroundPredicateArguments(List<DatabaseGroundPredicateArgument> groundPredicateArguments) {
        this.groundPredicateArguments = groundPredicateArguments;
    }
}
