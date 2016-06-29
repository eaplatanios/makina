package makina.optimization.constraint;

import makina.optimization.function.AbstractFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractInequalityConstraint extends AbstractFunction {
    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        AbstractInequalityConstraint that = (AbstractInequalityConstraint) other;

        if (!super.equals(that))
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        return super.hashCode();
    }
}
