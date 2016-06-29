package makina.learn.logic.grounding;

import makina.learn.logic.LogicManager;

/**
 * @author Emmanouil Antonios Platanios
 */
public enum GroundingMethod {
    IN_MEMORY {
        @Override
        public InMemoryLazyGrounding getGrounding(LogicManager logicManager) {
            return new InMemoryLazyGrounding(logicManager);
        }
    };

    public abstract InMemoryLazyGrounding getGrounding(LogicManager logicManager);
}
