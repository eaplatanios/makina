package org.platanios.learn.optimization.constraint;

import java.io.IOException;
import java.io.InputStream;

/**
 * This enumeration contains the different types of constraints that are supported. Each type also contains methods that can
 * be called to build constraints of the corresponding type.
 *
 * @author Dan Schwartz
 */
public enum ConstraintType {
    LinearEqualityConstraint {
        @Override
        public LinearEqualityConstraint buildConstraint(InputStream inputStream, boolean includeType) throws IOException {
            return org.platanios.learn.optimization.constraint.LinearEqualityConstraint.read(inputStream, includeType);
        }
    };

    public abstract AbstractConstraint buildConstraint(InputStream inputStream, boolean includeType) throws IOException;
}
