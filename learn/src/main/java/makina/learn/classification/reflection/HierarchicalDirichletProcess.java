
/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package makina.learn.classification.reflection;

/**
 *
 * @author akdubey
 */
import gnu.trove.iterator.TIntIterator;
import gnu.trove.set.hash.TIntHashSet;


public class HierarchicalDirichletProcess {

    int table_seated[][]; // Contains the table id of each item. indexed by group_id , item_location

    int count_seated_table[][]; // Contains the count of number of people seating at each table. indexed by group_id, table_id


    int tables_topic_id[][]; // which topic is assigned to which table. indexed by group_id, table_id

    int numberOfTablesPerTopic[]; // Count of tables assigned for each topic. Indexed by topic_id

    int numberOfTables; // Sum of all tables assigned ie sum of numberOfTablesPerTopic

    TIntHashSet available_topcis; // contains available topics

    TIntHashSet taken_topics; // contains taken topics

    TIntHashSet available_tables[]; // for each group list of available tables

    TIntHashSet taken_tables[]; // for each group list of taken tables

    TIntHashSet table_items[][]; //list of all items on a table, F otherwise, indexed by group_id, table_id, item_location

    double alpha; // higher level hyper-parameter
    double gamma; // lower level hyper_parameter

    public final int[] clusterTables;
    public final int[] clusterTopics;
    public final double[] clusterUnnormalizedProbabilities;

    public HierarchicalDirichletProcess(int N, int M, double alpha, double gamma) {
        int max_topic = N * M;
        table_seated = new int[N][M];
        count_seated_table = new int[N][M];
        tables_topic_id = new int[N][M];
        numberOfTablesPerTopic = new int[max_topic];
        numberOfTables = 0;
        available_topcis = new TIntHashSet();
        taken_topics = new TIntHashSet();
        for (int i = 0; i < max_topic; i++) {
            available_topcis.add(i);
        }
        available_tables = new TIntHashSet[N];
        taken_tables = new TIntHashSet[N];
        table_items = new TIntHashSet[N][M];

        for (int i = 0; i < N; i++) {
            available_tables[i] = new TIntHashSet();
            taken_tables[i] = new TIntHashSet();
            for (int j = 0; j < M; j++) {
                table_items[i][j] = new TIntHashSet();
                available_tables[i].add(j);
            }
        }
        this.alpha = alpha;
        this.gamma = gamma;

        clusterTables = new int[max_topic + M];
        clusterTopics = new int[max_topic + M];
        clusterUnnormalizedProbabilities = new double[max_topic + M];
    }

    /**
     * This function removes the assignment of table for the item given by the
     * two parameter and updates the relevant data structures
     *
     * @param group_id
     * @param item_location
     */
    public void remove_items_table_assignment(int group_id, int item_location) {
        int table_id = table_seated[group_id][item_location];
        int topic_id = tables_topic_id[group_id][table_id];

        table_seated[group_id][item_location] = -1; //remove alocation of table
        count_seated_table[group_id][table_id]--; //reduce count of assignment to table

        table_items[group_id][table_id].remove(item_location);//remove item location from the list containing all items on that table

        if (count_seated_table[group_id][table_id] == 0) { // if table becomes empty
            tables_topic_id[group_id][table_id] = -1;    //remove assignment of topic to table
            numberOfTablesPerTopic[topic_id]--;              // reduce count of number of tables to which topic is assigned to
            numberOfTables--;                          // total number of table assignment for topics
            available_tables[group_id].add(table_id);   // add table to list of available tables
            taken_tables[group_id].remove(table_id);    // remove table from list of taken tables
            if (numberOfTablesPerTopic[topic_id] == 0) {       // if the topic is no longer assigned to any table
                available_topcis.add(topic_id);         // add topic to list of available topics
                taken_topics.remove(topic_id);          // remove topic from list of taken topics
            }
        }
    }


    /**
     * This function adds assignment of a given table with id table_id and topic
     * topic_id and update the relevant data structures
     *
     * @param group_id
     * @param item_location
     * @param table_id
     * @param topic_id
     */
    public void add_items_table_assignment(int group_id, int item_location, int table_id, int topic_id) {

        table_seated[group_id][item_location] = table_id;
        count_seated_table[group_id][table_id]++;

        table_items[group_id][table_id].add(item_location);

        if (count_seated_table[group_id][table_id] == 1) {
            tables_topic_id[group_id][table_id] = topic_id;
            numberOfTablesPerTopic[topic_id]++;
            numberOfTables++;
            available_tables[group_id].remove(table_id);
            taken_tables[group_id].add(table_id);

            if (numberOfTablesPerTopic[topic_id] == 1) {
                available_topcis.remove(topic_id);
                taken_topics.add(topic_id);
            }
        }
    }


    /**
     * this function samples a table and its corresponding topic assignment for
     * an item
     *
     * @param group_id
     * @param item_location
     * @return
     */
    public int prob_table_assignment_for_item(int group_id, int item_location) {

        int total_positions = taken_tables[group_id].size() + taken_topics.size() + 1;
        if (total_positions > clusterUnnormalizedProbabilities.length) {
            total_positions = clusterUnnormalizedProbabilities.length;
        }

        TIntIterator it = taken_tables[group_id].iterator();
        int position = 0;
        int table_id;
        int topic_id;

        while (it.hasNext()) {
            table_id = it.next();
            topic_id = tables_topic_id[group_id][table_id];
            clusterUnnormalizedProbabilities[position] = count_seated_table[group_id][table_id];
            clusterTopics[position] = topic_id;
            clusterTables[position] = table_id;

            position++;
        }
        it = available_tables[group_id].iterator();
        if (!it.hasNext()) {
            return position;
        }
        table_id = it.next();
        it = taken_topics.iterator();
        while (it.hasNext() || Double.isNaN(Math.log(gamma))) {
            topic_id = it.next();
            clusterUnnormalizedProbabilities[position] = gamma * (numberOfTablesPerTopic[topic_id] / (alpha + numberOfTables));
            clusterTables[position] = table_id;
            clusterTopics[position] = topic_id;
            position++;
        }
        it = available_topcis.iterator();
        if (!it.hasNext() || Double.isNaN(Math.log(alpha))) {
            return position;
        }
        topic_id = it.next();
        clusterUnnormalizedProbabilities[position] = gamma * (alpha / (alpha + numberOfTables));
        clusterTables[position] = table_id;
        clusterTopics[position] = topic_id;
        position++;
        return position;
    }


    public int[] remove_tables_topic_assignment(int group_id, int table_id) {

        int topic_id = tables_topic_id[group_id][table_id];

        tables_topic_id[group_id][table_id] = -1;
        numberOfTablesPerTopic[topic_id]--;
        numberOfTables--;

        if (numberOfTablesPerTopic[topic_id] == 0) {
            available_topcis.add(topic_id);
            taken_topics.remove(topic_id);
        }

        return table_items[group_id][table_id].toArray();

    }

    public void add_tobles_topic_assignment(int group_id, int table_id, int topic_id) {
        tables_topic_id[group_id][table_id] = topic_id;
        numberOfTablesPerTopic[topic_id]++;
        numberOfTables++;

        if (numberOfTablesPerTopic[topic_id] == 1) {
            available_topcis.remove(topic_id);
            taken_topics.add(topic_id);
        }
    }

    public int prob_topic_assignment_for_table(int group_id, int table_id) {

        int position = 0;
        TIntIterator it = taken_topics.iterator();
        int topic_id;

        while (it.hasNext()) {
            topic_id = it.next();
            clusterUnnormalizedProbabilities[position] = numberOfTablesPerTopic[topic_id];
            clusterTopics[position] = topic_id;
            position++;
        }
        if (available_topcis.size() == 0 || Double.isNaN(Math.log(alpha))) {
            return position;
        }
        clusterUnnormalizedProbabilities[position] = alpha;
        it = available_topcis.iterator();
        clusterTopics[position] = it.next();
        position++;

        return position;

    }

    public int[] get_tables_taken(int group_id) {
        return taken_tables[group_id].toArray();
    }

    public int get_topic_table(int group_id, int table_id) {
        return tables_topic_id[group_id][table_id];
    }

    public int[] get_topics() {
        return taken_topics.toArray();
    }

    public TIntHashSet getTakenTopics() {
        return taken_topics;
    }

    public int getNumberOfTablesForTopic(int topicID) {
        return numberOfTablesPerTopic[topicID];
    }

    public int getNumberOfTables() {
        return numberOfTables;
    }
}
