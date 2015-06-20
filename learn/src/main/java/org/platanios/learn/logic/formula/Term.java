package org.platanios.learn.logic.formula;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class Term {
    long id;
    EntityType type;

    public Term(long id, EntityType type) {
        this.id = id;
        this.type = type;
    }

    public long getId() {
        return id;
    }

    public EntityType getType() {
        return type;
    }

    @Override
    public abstract String toString();
}
