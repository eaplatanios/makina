package org.platanios.learn.classification.reflection;

import gnu.trove.iterator.TIntIterator;
import gnu.trove.set.hash.TIntHashSet;

/**
 *
 * @author Emmanouil Antonios Platanios
 */
public class DirichletProcessPrior {
    private final double alpha;
    private final int clusterMemberCounts[];
    private final TIntHashSet freeClusters;
    private final TIntHashSet populatedClusters;
    private final int clusterIDs[];
    private final double clusterUnnormalizedProbabilities[];

    private int currentNumberOfClusters;
    
    public DirichletProcessPrior(double alpha, int maximumNumberOfClusters) {
        this.alpha = alpha;
        clusterMemberCounts = new int[maximumNumberOfClusters];
        freeClusters = new TIntHashSet();
        for (int clusterID = 0; clusterID < maximumNumberOfClusters; clusterID++)
            freeClusters.add(clusterID);
        populatedClusters = new TIntHashSet();
        clusterIDs = new int[maximumNumberOfClusters];
        clusterUnnormalizedProbabilities = new double[maximumNumberOfClusters];
        currentNumberOfClusters = 0;
    }

    public void addMemberToCluster(int clusterID) {
        if (++clusterMemberCounts[clusterID] == 1) {
            freeClusters.remove(clusterID);
            populatedClusters.add(clusterID);
        }
    }
    
    public void removeMemberFromCluster(int clusterID) {
        if (--clusterMemberCounts[clusterID] == 0) {
            freeClusters.add(clusterID);
            populatedClusters.remove(clusterID);
        }
    }
    
    public int computeClustersDistribution() {
        TIntIterator populatedClustersIterator = populatedClusters.iterator();
        currentNumberOfClusters = 0;
        while(populatedClustersIterator.hasNext()) {
            int cluster = populatedClustersIterator.next();
            clusterIDs[currentNumberOfClusters] = cluster;
            clusterUnnormalizedProbabilities[currentNumberOfClusters] = clusterMemberCounts[cluster];
            currentNumberOfClusters++;
        }
        if (freeClusters.size() == 0)
            return currentNumberOfClusters;
        clusterIDs[currentNumberOfClusters] = freeClusters.iterator().next();
        clusterUnnormalizedProbabilities[currentNumberOfClusters] = alpha;
        currentNumberOfClusters++;
        return currentNumberOfClusters;
    }

    public int getClusterID(int clusterPosition) {
        return clusterIDs[clusterPosition];
    }

    public double getClusterUnnormalizedProbability(int clusterPosition) {
        return clusterUnnormalizedProbabilities[clusterPosition];
    }
}
