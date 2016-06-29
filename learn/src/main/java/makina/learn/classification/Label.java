package makina.learn.classification;

import com.google.common.base.Objects;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Label {
    private final String name;

    public Label(String name) {
        this.name = name;
    }

    public String name() {
        return name;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        Label that = (Label) other;

        return Objects.equal(name, that.name);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(name);
    }

    @Override
    public String toString() {
        return name;
    }
}
