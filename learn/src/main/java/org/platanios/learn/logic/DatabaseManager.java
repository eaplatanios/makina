package org.platanios.learn.logic;

import org.hibernate.*;
import org.hibernate.boot.registry.StandardServiceRegistryBuilder;
import org.hibernate.cfg.Configuration;
import org.hibernate.criterion.Projections;
import org.hibernate.criterion.Restrictions;
import org.hibernate.service.ServiceRegistry;
import org.platanios.learn.logic.formula.*;
import org.platanios.learn.logic.grounding.GroundFormula;
import org.platanios.learn.logic.grounding.GroundFormulaPredicate;
import org.platanios.learn.logic.grounding.GroundPredicate;
import org.platanios.learn.logic.grounding.GroundPredicateArgument;

import java.util.*;
import java.util.stream.Collectors;

/**
 * TODO: Change the code so that we do not keep opening and closing sessions, but rather use the getCurrentSession() method.
 *
 * @author Emmanouil Antonios Platanios
 */
public class DatabaseManager {
    // Connection Settings
    private final String driverClass;
    private final String connectionURL;
    private final String username;
    private final String password;
    private final int connectionPoolSize;
    private final String dialect;

    // Other Settings
    private final String currentSessionContextClass;
    private final boolean enableLazyLoadWithNoTransaction;
    private final boolean showSQL;
    private final boolean newGeneratorMappings;
    private final int jdbcBatchSize;
    private final boolean orderInserts;
    private final boolean orderUpdates;
    private final boolean batchVersionedData;
    private final String hbm2ddlAuto;

    private SessionFactory sessionFactory;

    public static class Builder {
        private String driverClass = "org.mariadb.jdbc.Driver";
        private String connectionURL = "jdbc:mariadb://localhost/learn_logic";
        private String username = "root";
        private String password = "";
        private int connectionPoolSize = 1;
        private String dialect = "org.hibernate.dialect.MySQLDialect";
        private String currentSessionContextClass = "thread";
        private boolean enableLazyLoadWithNoTransaction = false;
        private boolean showSQL = false;
        private boolean newGeneratorMappings = true;
        private int jdbcBatchSize = 20;
        private boolean orderInserts = true;
        private boolean orderUpdates = true;
        private boolean batchVersionedData = true;
        private String hbm2ddlAuto = "create-drop";

        public Builder() {

        }

        public Builder driverClass(String driverClass) {
            this.driverClass = driverClass;
            return this;
        }

        public Builder connectionURL(String connectionURL) {
            this.connectionURL = connectionURL;
            return this;
        }

        public Builder username(String username) {
            this.username = username;
            return this;
        }

        public Builder password(String password) {
            this.password = password;
            return this;
        }

        public Builder connectionPoolSize(int connectionPoolSize) {
            this.connectionPoolSize = connectionPoolSize;
            return this;
        }

        public Builder dialect(String dialect) {
            this.dialect = dialect;
            return this;
        }

        public Builder currentSessionContextClass(String currentSessionContextClass) {
            this.currentSessionContextClass = currentSessionContextClass;
            return this;
        }

        public Builder enableLazyLoadWithNoTransaction(boolean enableLazyLoadWithNoTransaction) {
            this.enableLazyLoadWithNoTransaction = enableLazyLoadWithNoTransaction;
            return this;
        }

        public Builder showSQL(boolean showSQL) {
            this.showSQL = showSQL;
            return this;
        }

        public Builder newGeneratorMappings(boolean newGeneratorMappings) {
            this.newGeneratorMappings = newGeneratorMappings;
            return this;
        }

        public Builder jdbcBatchSize(int jdbcBatchSize) {
            this.jdbcBatchSize = jdbcBatchSize;
            return this;
        }

        public Builder orderInserts(boolean orderInserts) {
            this.orderInserts = orderInserts;
            return this;
        }

        public Builder orderUpdates(boolean orderUpdates) {
            this.orderUpdates = orderUpdates;
            return this;
        }

        public Builder batchVersionedData(boolean batchVersionedData) {
            this.batchVersionedData = batchVersionedData;
            return this;
        }

        public Builder hbm2ddlAuto(String hbm2ddlAuto) {
            this.hbm2ddlAuto = hbm2ddlAuto;
            return this;
        }

        public DatabaseManager build() {
            return new DatabaseManager(this);
        }
    }

    private DatabaseManager(Builder builder) {
        driverClass = builder.driverClass;
        connectionURL = builder.connectionURL;
        username = builder.username;
        password = builder.password;
        connectionPoolSize = builder.connectionPoolSize;
        dialect = builder.dialect;
        currentSessionContextClass = builder.currentSessionContextClass;
        enableLazyLoadWithNoTransaction = builder.enableLazyLoadWithNoTransaction;
        showSQL = builder.showSQL;
        newGeneratorMappings = builder.newGeneratorMappings;
        jdbcBatchSize = builder.jdbcBatchSize;
        orderInserts = builder.orderInserts;
        orderUpdates = builder.orderUpdates;
        batchVersionedData = builder.batchVersionedData;
        hbm2ddlAuto = builder.hbm2ddlAuto;

        buildSessionFactory();
    }

    public void buildSessionFactory() {
        Configuration configuration =
                new Configuration()
                        .setProperty("hibernate.connection.driver_class", driverClass)
                        .setProperty("hibernate.connection.url", connectionURL)
                        .setProperty("hibernate.connection.username", username)
                        .setProperty("hibernate.connection.password", password)
                        .setProperty("hibernate.connection.pool_size", Integer.toString(connectionPoolSize))
                        .setProperty("hibernate.dialect", dialect)
                        .setProperty("hibernate.current_session_context_class", currentSessionContextClass)
                        .setProperty("hibernate.enable_lazy_load_no_trans",
                                     Boolean.toString(enableLazyLoadWithNoTransaction))
                        .setProperty("hibernate.show_sql", Boolean.toString(showSQL))
                        .setProperty("hibernate.id.new_generator_mappings", Boolean.toString(newGeneratorMappings))
                        .setProperty("hibernate.jdbc.batch_size", Integer.toString(jdbcBatchSize))
                        .setProperty("hibernate.order_inserts", Boolean.toString(orderInserts))
                        .setProperty("hibernate.order_updates", Boolean.toString(orderUpdates))
                        .setProperty("hibernate.jdbc.batch_versioned_data", Boolean.toString(batchVersionedData))
                        .setProperty("hibernate.hbm2ddl.auto", hbm2ddlAuto)
                        .addAnnotatedClass(EntityType.class)
                        .addAnnotatedClass(EntityTypeValue.class)
                        .addAnnotatedClass(Predicate.class)
                        .addAnnotatedClass(PredicateArgumentType.class)
                        .addAnnotatedClass(GroundPredicate.class)
                        .addAnnotatedClass(GroundPredicateArgument.class)
                        .addAnnotatedClass(GroundFormula.class)
                        .addAnnotatedClass(GroundFormulaPredicate.class);
        ServiceRegistry serviceRegistry = new StandardServiceRegistryBuilder()
                .applySettings(configuration.getProperties())
                .build();
        sessionFactory = configuration.buildSessionFactory(serviceRegistry);
    }

    public void closeSessionFactory() {
        sessionFactory.close();
    }

    private void insertObject(Object object) {
        Session session = sessionFactory.openSession();
        Transaction transaction = session.beginTransaction();
        session.save(object);
        transaction.commit();
        session.close();
    }

    private void insertObjects(List<?> objects) {
        Session session = sessionFactory.openSession();
        Transaction transaction = session.beginTransaction();
        int numberOfObjectsSaved = 0;
        for (Object object : objects) {
            session.save(object);
            if (++numberOfObjectsSaved % jdbcBatchSize == 0) {
                session.flush();
                session.clear();
            }
        }
        transaction.commit();
        session.close();
    }

    private void fastInsertObject(Object object) {
        StatelessSession session = sessionFactory.openStatelessSession();
        Transaction transaction = session.beginTransaction();
        session.insert(object);
        transaction.commit();
        session.close();
    }

    private void fastInsertObjects(List<?> objects) {
        StatelessSession session = sessionFactory.openStatelessSession();
        Transaction transaction = session.beginTransaction();
        objects.forEach(session::insert);
        transaction.commit();
        session.close();
    }

    private void updateObject(Object object) {
        Session session = sessionFactory.openSession();
        Transaction transaction = session.beginTransaction();
        session.update(object);
        transaction.commit();
        session.close();
    }

    private void updateObjects(List<?> objects) {
        Session session = sessionFactory.openSession();
        Transaction transaction = session.beginTransaction();
        int numberOfObjectsUpdated = 0;
        for (Object object : objects) {
            session.update(object);
            if (++numberOfObjectsUpdated % jdbcBatchSize == 0) {
                session.flush();
                session.clear();
            }
        }
        transaction.commit();
        session.close();
    }

    private void fastUpdateObject(Object object) {
        StatelessSession session = sessionFactory.openStatelessSession();
        Transaction transaction = session.beginTransaction();
        session.update(object);
        transaction.commit();
        session.close();
    }

    private void fastUpdateObjects(List<?> objects) {
        StatelessSession session = sessionFactory.openStatelessSession();
        Transaction transaction = session.beginTransaction();
        objects.forEach(session::update);
        transaction.commit();
        session.close();
    }

    private Object getObject(Class<?> type, long id) {
        Session session = sessionFactory.openSession();
        Object object = session.get(type, id);
        session.close();
        return object;
    }

    private Object getObjectProxy(Class<?> type, long id) {
        Session session = sessionFactory.openSession();
        Object object = session.load(type, id);
        session.close();
        return object;
    }

    private boolean checkIfObjectExists(Class<?> type, long id) {
        return getObject(type, id) != null;
    }

    private long getNumberOfRows(Class<?> type) {
        Session session = sessionFactory.openSession();
        long numberOfRows = (long) session.createCriteria(type)
                .setProjection(Projections.rowCount())
                .uniqueResult();
        session.close();
        return numberOfRows;
    }

    public EntityType addEntityType(Set<Long> allowedValues) {
        EntityType entityType = new EntityType();
        entityType.setAllowedValues(
                allowedValues.stream()
                        .map(value -> new EntityTypeValue(entityType, value))
                        .collect(Collectors.toSet())
        );
        insertObject(entityType);
        return entityType;
    }

    public EntityType addEntityType(String name, Set<Long> allowedValues) {
        EntityType entityType = new EntityType(name);
        entityType.setAllowedValues(
                allowedValues.stream()
                        .map(value -> new EntityTypeValue(entityType, value))
                        .collect(Collectors.toSet())
        );
        insertObject(entityType);
        return entityType;
    }

    public EntityType getEntityType(long id) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(EntityType.class);
        criteria.setFetchMode("allowedValues", FetchMode.JOIN);
        criteria.add(Restrictions.eq("id", id));
        EntityType entityType = (EntityType) criteria.uniqueResult();
        session.close();
        return entityType;
    }

    public EntityType getEntityType(String name) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(EntityType.class);
        criteria.setFetchMode("allowedValues", FetchMode.JOIN);
        criteria.add(Restrictions.eq("name", name));
        EntityType entityType = (EntityType) criteria.uniqueResult();
        session.close();
        return entityType;
    }

    public Set<EntityTypeValue> getEntityTypeAllowedValues(long id) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(EntityType.class);
        criteria.setFetchMode("allowedValues", FetchMode.JOIN);
        criteria.add(Restrictions.eq("id", id));
        EntityType entityType = (EntityType) criteria.uniqueResult();
        session.close();
        return entityType.getAllowedValues();
    }

    public Predicate addPredicate(boolean closed) {
        return addPredicate((String) null, closed);
    }

    public Predicate addPredicate(String name, boolean closed) {
        Predicate predicate = new Predicate();
        predicate.setName(name);
        predicate.setClosed(closed);
        insertObject(predicate);
        return predicate;
    }

    public Predicate addPredicate(List<EntityType> argumentTypes, boolean closed) {
        return addPredicate(null, argumentTypes, closed);
    }

    public Predicate addPredicate(String name, List<EntityType> argumentTypes, boolean closed) {
        Predicate predicate = new Predicate();
        predicate.setName(name);
        predicate.setClosed(closed);
        List<PredicateArgumentType> predicateArgumentTypes = new ArrayList<>();
        for (int argumentIndex = 0; argumentIndex < argumentTypes.size(); argumentIndex++)
            predicateArgumentTypes.add(
                    new PredicateArgumentType(predicate,
                                              argumentIndex,
                                              argumentTypes.get(argumentIndex))
            );
        predicate.setArgumentTypes(predicateArgumentTypes);
        insertObject(predicate);
        return predicate;
    }

    public Predicate getPredicate(long id) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(Predicate.class);
        criteria.setFetchMode("argumentTypes", FetchMode.JOIN);
        criteria.add(Restrictions.eq("id", id));
        Predicate predicate = (Predicate) criteria.uniqueResult();
        session.close();
        return predicate;
    }

    public Predicate getPredicate(String name) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(Predicate.class);
        criteria.setFetchMode("argumentTypes", FetchMode.JOIN);
        criteria.add(Restrictions.eq("name", name));
        Predicate predicate = (Predicate) criteria.uniqueResult();
        session.close();
        return predicate;
    }

    public Predicate getDatabasePredicate(long id) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(Predicate.class);
        criteria.setFetchMode("argumentTypes", FetchMode.JOIN);
        criteria.add(Restrictions.eq("id", id));
        Predicate predicate = (Predicate) criteria.uniqueResult();
        session.close();
        return predicate;
    }

    public Predicate getDatabasePredicate(String name) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(Predicate.class);
        criteria.setFetchMode("argumentTypes", FetchMode.JOIN);
        criteria.add(Restrictions.eq("name", name));
        Predicate predicate = (Predicate) criteria.uniqueResult();
        session.close();
        return predicate;
    }

    @SuppressWarnings("unchecked")
    public List<Predicate> getClosedPredicates() {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(Predicate.class);
        criteria.add(Restrictions.eq("closed", true));
        List<Predicate> closedPredicates = ((List<Predicate>) criteria.list());
        session.close();
        return closedPredicates;
    }

    public GroundPredicate addGroundPredicate(long predicateId,
                                              List<Long> variablesAssignment,
                                              Double value) {
        Predicate predicate = (Predicate) getObject(Predicate.class, predicateId);
        GroundPredicate groundPredicate = new GroundPredicate(predicate, value);
        insertObject(groundPredicate);
        List<GroundPredicateArgument> groundPredicateArguments = new ArrayList<>();
        for (int argumentIndex = 0; argumentIndex < variablesAssignment.size(); argumentIndex++)
            groundPredicateArguments.add(new GroundPredicateArgument(
                    predicate,
                    groundPredicate,
                    argumentIndex,
                    variablesAssignment.get(argumentIndex)
            ));
        groundPredicate.setGroundPredicateArguments(groundPredicateArguments);
        updateObject(groundPredicate);
        return groundPredicate;
    }

    public GroundPredicate addGroundPredicate(GroundPredicate groundPredicate) {
        insertObject(groundPredicate);
        return groundPredicate;
    }

    public List<GroundPredicate> addGroundPredicates(List<GroundPredicate> groundPredicates) {
        insertObjects(groundPredicates);
        return groundPredicates;
    }

    @SuppressWarnings("unchecked")
    public boolean checkIfGroundPredicateExists(long predicateId, List<Long> variablesAssignment) {
        StringBuilder hqlQuery = new StringBuilder("select count(*) from ");
        for (int argumentId = 0; argumentId < variablesAssignment.size(); argumentId++) {
            hqlQuery.append("GroundPredicateArgument ")
                    .append("argument_").append(argumentId);
            if (argumentId < variablesAssignment.size() - 1)
                hqlQuery.append(", ");
        }
        hqlQuery.append(" where ");
        hqlQuery.append("argument_0.predicate.id = ").append(predicateId);
        for (int argumentId = 0; argumentId < variablesAssignment.size(); argumentId++) {
            if (argumentId != 0)
                hqlQuery.append(" and argument_0.groundPredicate.id = ")
                        .append("argument_").append(argumentId)
                        .append(".groundPredicate.id");
            hqlQuery.append(" and argument_").append(argumentId)
                    .append(".argumentIndex = ").append(argumentId);
            hqlQuery.append(" and argument_").append(argumentId)
                    .append(".argumentValue = ").append(variablesAssignment.get(argumentId));
        }
        Session session = sessionFactory.openSession();
        Query query = session.createQuery(hqlQuery.toString());
        boolean result = (long) query.uniqueResult() > 0;
        session.close();
        return result;
    }

    public long getNumberOfGroundPredicates() {
        return getNumberOfRows(GroundPredicate.class);
    }

    @SuppressWarnings("unchecked")
    public List<GroundPredicate> getGroundPredicates() {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(GroundPredicate.class);
        criteria.setFetchMode("predicate", FetchMode.JOIN);
        criteria.setFetchMode("groundPredicateArguments", FetchMode.JOIN);
        List<GroundPredicate> groundPredicates = ((List<GroundPredicate>) criteria.list());
        session.close();
        return groundPredicates;
    }

    @SuppressWarnings("unchecked")
    public GroundPredicate getGroundPredicate(long id) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(GroundPredicate.class);
        criteria.setFetchMode("groundPredicateArguments", FetchMode.JOIN);
        criteria.add(Restrictions.eq("id", id));
        GroundPredicate groundPredicate = (GroundPredicate) criteria.uniqueResult();
        session.close();
        return groundPredicate;
    }

    @SuppressWarnings("unchecked")
    public GroundPredicate getGroundPredicate(long predicateId, List<Long> variablesAssignment) {
        StringBuilder hqlQuery = new StringBuilder("from ");
        for (int argumentId = 0; argumentId < variablesAssignment.size(); argumentId++) {
            hqlQuery.append("GroundPredicateArgument ")
                    .append("argument_").append(argumentId);
            if (argumentId < variablesAssignment.size() - 1)
                hqlQuery.append(", ");
        }
        hqlQuery.append(" where ");
        hqlQuery.append("argument_0.predicate.id = ").append(predicateId);
        for (int argumentId = 0; argumentId < variablesAssignment.size(); argumentId++) {
            if (argumentId != 0)
                hqlQuery.append(" and argument_0.groundPredicate.id = ")
                        .append("argument_").append(argumentId)
                        .append(".groundPredicate.id");
            hqlQuery.append(" and argument_").append(argumentId)
                    .append(".argumentIndex = ").append(argumentId);
            hqlQuery.append(" and argument_").append(argumentId)
                    .append(".argumentValue = ").append(variablesAssignment.get(argumentId));
        }
        Session session = sessionFactory.openSession();
        Query query = session.createQuery(hqlQuery.toString());
        GroundPredicateArgument argument = ((GroundPredicateArgument) ((Object[]) query.uniqueResult())[0]);
        GroundPredicate groundPredicate = argument.getGroundPredicate();
        session.close();
        return groundPredicate;
    }

    @SuppressWarnings("unchecked")
    public Double getPredicateAssignmentTruthValue(Predicate predicate, List<Long> variablesAssignment, Logic logic) {
        StringBuilder hqlQuery = new StringBuilder("from ");
        for (int argumentId = 0; argumentId < variablesAssignment.size(); argumentId++) {
            hqlQuery.append("GroundPredicateArgument ")
                    .append("argument_").append(argumentId);
            if (argumentId < variablesAssignment.size() - 1)
                hqlQuery.append(", ");
        }
        hqlQuery.append(" where ");
        hqlQuery.append("argument_0.predicate.id = ").append(predicate.getId());
        for (int argumentId = 0; argumentId < variablesAssignment.size(); argumentId++) {
            if (argumentId != 0)
                hqlQuery.append(" and argument_0.groundPredicate.id = ")
                        .append("argument_").append(argumentId)
                        .append(".groundPredicate.id");
            hqlQuery.append(" and argument_").append(argumentId)
                    .append(".argumentIndex = ").append(argumentId);
            hqlQuery.append(" and argument_").append(argumentId)
                    .append(".argumentValue = ").append(variablesAssignment.get(argumentId));
        }
        StatelessSession session = sessionFactory.openStatelessSession();
        Query query = session.createQuery(hqlQuery.toString());
        Object[] uniqueResult = (Object[]) query.uniqueResult();
        if (uniqueResult != null) {
            GroundPredicateArgument argument = ((GroundPredicateArgument) uniqueResult[0]);
            GroundPredicate groundPredicate = argument.getGroundPredicate();
            session.close();
            return groundPredicate.getValue();
        } else if (!getDatabasePredicate(predicate.getId()).getClosed()) {
            session.close();
            return null;
        } else {
            session.close();
            return logic.falseValue();
        }
    }

    @SuppressWarnings("unchecked")
    public PartialGroundedFormula getMatchingGroundPredicates(List<Atom> atoms, Logic logic) {
        StringBuilder hqlQuery = new StringBuilder("from ");
        for (int atomId = 0; atomId < atoms.size(); atomId++) {
            for (int argumentId = 0; argumentId < atoms.get(atomId).getOrderedVariables().size(); argumentId++) {
                hqlQuery.append("GroundPredicateArgument ")
                        .append("atom_").append(atomId)
                        .append("_argument_").append(argumentId);
                if (atomId < atoms.size() - 1
                        || argumentId != atoms.get(atomId).getOrderedVariables().size() - 1)
                    hqlQuery.append(", ");
            }
        }
        hqlQuery.append(" where ");
        for (int atomId = 0; atomId < atoms.size(); atomId++) {
            if (atomId != 0)
                hqlQuery.append(" and ");
            hqlQuery.append("atom_").append(atomId)
                    .append("_argument_0.predicate.id = ")
                    .append(atoms.get(atomId).getPredicate().getId());
            for (int argumentId = 0; argumentId < atoms.get(atomId).getOrderedVariables().size(); argumentId++) {
                if (argumentId != 0)
                    hqlQuery.append(" and atom_").append(atomId)
                            .append("_argument_0.groundPredicate.id = ")
                            .append("atom_").append(atomId)
                            .append("_argument_").append(argumentId)
                            .append(".groundPredicate.id");
                hqlQuery.append(" and atom_").append(atomId)
                        .append("_argument_").append(argumentId)
                        .append(".argumentIndex = ").append(argumentId);
                Variable currentVariable = atoms.get(atomId).getOrderedVariables().get(argumentId);
                for (int innerAtomId = atomId + 1; innerAtomId < atoms.size(); innerAtomId++) {
                    List<Variable> innerVariables = atoms.get(innerAtomId).getOrderedVariables();
                    for (int innerArgumentId = 0; innerArgumentId < innerVariables.size(); innerArgumentId++)
                        if (currentVariable.getId() == innerVariables.get(innerArgumentId).getId())
                            hqlQuery.append(" and atom_").append(atomId)
                                    .append("_argument_").append(argumentId)
                                    .append(".argumentValue = ")
                                    .append("atom_").append(innerAtomId)
                                    .append("_argument_").append(innerArgumentId)
                                    .append(".argumentValue");
                }
            }
        }
        Session session = sessionFactory.openSession();
        Query query = session.createQuery(hqlQuery.toString());
        List<Object[]> resultList = query.list();
        PartialGroundedFormula partialGroundedFormula = new PartialGroundedFormula();
        for (Object[] result : resultList) {
            partialGroundedFormula.addFormulaUnobservedVariableIndicator(false);
            List<GroundPredicate> groundFormula = new ArrayList<>();
            List<Double> groundFormulaPartTruthValues = new ArrayList<>();
            Set<Long> seenVariables = new HashSet<>();
            int resultIndex = 0;
            for (Atom atom : atoms) {
                GroundPredicate currentGroundPredicate =
                        ((GroundPredicateArgument) result[resultIndex]).getGroundPredicate();
                groundFormula.add(currentGroundPredicate);
                Double truthValue = currentGroundPredicate.getValue();
                groundFormulaPartTruthValues.add(truthValue == null ? logic.falseValue() : logic.negation(truthValue));
                for (int argumentId = 0; argumentId < atom.getOrderedVariables().size(); argumentId++) {
                    Long currentVariableId = atom.getOrderedVariables().get(argumentId).getId();
                    if (!seenVariables.contains(currentVariableId)) {
                        partialGroundedFormula.addGroundVariable(
                                currentVariableId,
                                ((GroundPredicateArgument) result[resultIndex]).getArgumentValue()
                        );
                        seenVariables.add(currentVariableId);
                    }
                    resultIndex++;
                }
            }
            partialGroundedFormula.addGroundFormula(groundFormula);
            partialGroundedFormula.addGroundFormulaTruthValue(logic.disjunction(groundFormulaPartTruthValues));
        }
        session.close();
        return partialGroundedFormula;
    }

    public long getNumberOfEntityTypes() {
        return getNumberOfRows(EntityType.class);
    }

    public boolean checkIfVariableTypeIDExists(long id) {
        return checkIfObjectExists(EntityType.class, id);
    }

    public class PartialGroundedFormula {
        Map<Long, List<Long>> groundVariables = new HashMap<>(); // Maps from variable ID to list of grounded values -- the list is ordered in the same way as the groundedFormula list.
        List<List<GroundPredicate>> groundFormula = new ArrayList<>(); // List of groundings -- each grounding is a set of grounded predicates representing the formula terms.
        List<Double> groundFormulaTruthValues = new ArrayList<>();
        List<Boolean> formulaUnobservedVariableIndicators = new ArrayList<>();

        public PartialGroundedFormula() { }

        public void addGroundVariable(Long variableId, Long value) {
            if (!groundVariables.containsKey(variableId))
                groundVariables.put(variableId, new ArrayList<>());
            groundVariables.get(variableId).add(value);
        }

        public void addGroundFormula(List<GroundPredicate> groundFormula) {
            this.groundFormula.add(groundFormula);
        }

        public void addGroundFormulaTruthValue(Double truthValue) {
            groundFormulaTruthValues.add(truthValue);
        }

        public void addFormulaUnobservedVariableIndicator(boolean unobservedVariableIndicator) {
            formulaUnobservedVariableIndicators.add(unobservedVariableIndicator);
        }

        public Map<Long, List<Long>> getGroundVariables() {
            return groundVariables;
        }

        public List<List<GroundPredicate>> getGroundFormula() {
            return groundFormula;
        }

        public List<Double> getGroundFormulaTruthValues() {
            return groundFormulaTruthValues;
        }

        public List<Boolean> getFormulaUnobservedVariableIndicators() {
            return formulaUnobservedVariableIndicators;
        }
    }
}
