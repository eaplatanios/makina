package makina.learn.logic.formula;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Variable extends Term {
    private final String name;

    public Variable(long id, EntityType type) {
        super(id, type);
        this.name = null;
    }

    public Variable(long id, String name, EntityType type) {
        super(id, type);
        this.name = name;
    }

    public String getName() {
        return name;
    }

    @Override
    public String toString() {
        if (this.name != null)
            return name;
        else
            return Long.toString(id);
    }
}
