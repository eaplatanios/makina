/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package org.platanios.learn.classification.reflection;

import gnu.trove.iterator.TIntIterator;
import gnu.trove.set.hash.TIntHashSet;

/**
 *
 * @author akdubey
 */
public class FastDPPrior {
    double alpha;
    int max_topic_id;
    
    int counts[];
    int total_cnt;
    
    
    TIntHashSet available_topcis; // contains available topics
    
    TIntHashSet taken_topics; // contains taken topics
    
    public DS_DP pdf[];
    
    public FastDPPrior(double alpha, int max_topic){
        this.alpha = alpha;
        this.max_topic_id = max_topic;
        
        counts = new int[max_topic];
        total_cnt=0;
        
        available_topcis = new TIntHashSet();
        taken_topics = new TIntHashSet();
        pdf = new DS_DP[max_topic];
        for(int i=0; i<max_topic;i++){
            available_topcis.add(i);
            pdf[i] = new DS_DP();
        }
        
    }
    
    void remove_topic_assignment(int topic_id){
        counts[topic_id]--;
        total_cnt--;
        if(counts[topic_id]==0){
            available_topcis.add(topic_id);
            taken_topics.remove(topic_id);
        }
    }
    
    void add_topic_assingment(int topic_id){
        counts[topic_id]++;
        total_cnt++;
        if(counts[topic_id]==1){
            available_topcis.remove(topic_id);
            taken_topics.add(topic_id);
        }
    }
    
    int prob_topics(){
        int total_positions = taken_topics.size() + 1;
        if(total_positions >= max_topic_id){
            total_positions = max_topic_id;
        }
        TIntIterator it = taken_topics.iterator();
        int position =0;
        int topic_id;
        while(it.hasNext()){
            topic_id = it.next();
            pdf[position].prob = counts[topic_id];
            pdf[position].topic = topic_id;
            position++;
        }
        if(available_topcis.size()==0){
            return position;
        }
        it = available_topcis.iterator();
        topic_id = it.next();
        pdf[position].prob = alpha;
        pdf[position].topic = topic_id;
        position++;
        
        return position;
    }
    
}
