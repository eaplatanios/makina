/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.platanios.learn.classification.reflection;

import gnu.trove.map.hash.TIntIntHashMap;
import org.apache.commons.math3.random.RandomDataGenerator;

import java.util.List;
import java.util.Random;

import static org.apache.commons.math3.special.Beta.logBeta;

/**
 *
 * @author avinava
 */
public class ErrorEstimationDomainsHDPNew {

    boolean f[][][];    //indexed by domain_id, classifier_id, example_number
    boolean l[][];      // indexed by domain_id, classifier_id

    FastHDPPrior hdp;
    int z[][];          //indexed by domain_id, classifier_id

    double alphap = 1;
    double betap = 1;

    double p[];     //indexed by domain id

//    double e[][];   //indexed by domain_id, classifier_id
    double error_rate[]; // indexed by topic_id

    double alphae = 1;
    double betae = 10;

    int li_cnt[][];    // indexed by domain_id, possitive/negative prediction
    int sum_li[];       // indexed by domain_id
    int err_cnt[][];    // indexed by topic, possitive/negative error
    int sum_err[];      // indexed by topic id

    int num_domain;     // total number of domain
    int num_classifier; //total number of classifier
    int num_example[];   // number of examples in each domain

    Random rm;
    private final RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
    int disagreement[][];   //indexed by domain_id, classifier_id

    public ErrorEstimationDomainsHDPNew(boolean f[][][], double alphap, double betap, double alphae, double betae, int num_example[],
            int num_domain, int num_classifier, double alpha, double gamma) {
        this.f = f;
        this.l = new boolean[num_domain][];
        for (int p = 0; p < num_domain; p++) {
            l[p] = new boolean[num_example[p]];
        }

        this.alphap = alphap;
        this.betap = betap;
        this.alphae = alphae;
        this.betae = betae;
        this.num_classifier = num_classifier;
        this.num_domain = num_domain;
        this.num_example = num_example;
        this.hdp = new FastHDPPrior(num_domain, num_classifier, alpha, gamma);
        z = new int[num_domain][num_classifier];
        p = new double[num_domain];
//        e = new double[num_domain][num_classifier];

        li_cnt = new int[num_domain][2];
        sum_li = new int[num_domain];

        err_cnt = new int[num_domain * num_classifier][2];
        sum_err = new int[num_domain * num_classifier];

        error_rate = new double[num_domain * num_classifier];

        rm = new Random(1983473);
        disagreement = new int[num_domain][num_classifier];
        initialize_for_sampling();
    }

    public ErrorEstimationDomainsHDPNew(List<boolean[][]> f, double alpha, double gamma) {
        this.num_domain = f.size();
        this.num_classifier = f.get(0)[0].length;
        this.num_example = new int[num_domain];
        for (int d = 0; d < num_domain; d++) {
            num_example[d] = f.get(d).length;
        }
        this.f = new boolean[num_domain][num_classifier][];
        this.l = new boolean[num_domain][];
        for (int d = 0; d < num_domain; d++) {
            l[d] = new boolean[f.get(d).length];
            for (int j = 0; j < num_classifier; j++) {
                this.f[d][j] = new boolean[f.get(d).length];
                for (int i = 0; i < num_example[d]; i++) {
                    this.f[d][j][i] = f.get(d)[i][j];
                }
            }
        }
        this.hdp = new FastHDPPrior(num_domain, num_classifier, alpha, gamma);
        z = new int[num_domain][num_classifier];
        p = new double[num_domain];
//        e = new double[num_domain][num_classifier];

        li_cnt = new int[num_domain][2];
        sum_li = new int[num_domain];

        err_cnt = new int[num_domain * num_classifier][2];
        sum_err = new int[num_domain * num_classifier];

        rm = new Random(1983473);
        disagreement = new int[num_domain][num_classifier];
        error_rate = new double[num_domain * num_classifier];
        initialize_for_sampling();

    }
    
    public ErrorEstimationDomainsHDPNew(List<boolean[][]> f, double alpha, double gamma, double []error_rate, int licnt[][]) {
        this.num_domain = f.size();
        this.num_classifier = f.get(0)[0].length;
        this.num_example = new int[num_domain];
        for (int d = 0; d < num_domain; d++) {
            num_example[d] = f.get(d).length;
        }
        this.f = new boolean[num_domain][num_classifier][];
        this.l = new boolean[num_domain][];
        for (int d = 0; d < num_domain; d++) {
            l[d] = new boolean[f.get(d).length];
            for (int j = 0; j < num_classifier; j++) {
                this.f[d][j] = new boolean[f.get(d).length];
                for (int i = 0; i < num_example[d]; i++) {
                    this.f[d][j][i] = f.get(d)[i][j];
                }
            }
        }
        this.hdp = new FastHDPPrior(num_domain, num_classifier, alpha, gamma);
        z = new int[num_domain][num_classifier];
        p = new double[num_domain];
//        e = new double[num_domain][num_classifier];

        li_cnt = licnt;
        sum_li = new int[num_domain];

        err_cnt = new int[num_domain * num_classifier][2];
        sum_err = new int[num_domain * num_classifier];

        rm = new Random(1983473);
        disagreement = new int[num_domain][num_classifier];
        this.error_rate = error_rate;
        initialize_for_sampling();
    }

    public void initialize_for_sampling() {
        for (int domain_id = 0; domain_id < num_domain; domain_id++) {
            for (int example_id = 0; example_id < num_example[domain_id]; example_id++) {
                int num_possitive = 0;
                int num_negative = 0;
                for (int classifier_id = 0; classifier_id < num_classifier; classifier_id++) {
                    num_possitive += f[domain_id][classifier_id][example_id] ? 1 : 0;
                    num_negative += f[domain_id][classifier_id][example_id] ? 0 : 1;
                }
                l[domain_id][example_id] = (num_possitive > num_negative);
                int lid = (l[domain_id][example_id]) ? 1 : 0;
                li_cnt[domain_id][lid]++;
                sum_li[domain_id]++;
                for (int classifier_id = 0; classifier_id < num_classifier; classifier_id++) {
                    disagreement[domain_id][classifier_id] += (f[domain_id][classifier_id][example_id] != l[domain_id][example_id]) ? 1 : 0;
                }
            }
        }

        for (int domain_id = 0; domain_id < num_domain; domain_id++) {
            for (int classifier_id = 0; classifier_id < num_classifier; classifier_id++) {
                z[domain_id][classifier_id] = 0;
                hdp.add_items_table_assignment(domain_id, classifier_id, 0, 0);
                err_cnt[0][0] += disagreement[domain_id][classifier_id];
                err_cnt[0][1] += (num_example[domain_id] - disagreement[domain_id][classifier_id]);
                sum_err[0] += num_example[domain_id];
            }
        }

    }

    public int[] update_before_sampling_tables_topic(int domain_id, int table_id) {
        int cls_ids[] = hdp.remove_tables_topic_assignment(domain_id, table_id);
        for (int classifier_id : cls_ids) {
            int topic_id = z[domain_id][classifier_id];
            err_cnt[topic_id][0] -= disagreement[domain_id][classifier_id];
            err_cnt[topic_id][1] -= (num_example[domain_id] - disagreement[domain_id][classifier_id]);
            sum_err[topic_id] -= num_example[domain_id];
        }
        return cls_ids;
    }

    public void update_after_sampling_tables_topic(int domain_id, int table_id, int topic_id, int cls_ids[]) {
        hdp.add_tobles_topic_assignment(domain_id, table_id, topic_id);
        for (int classifier_id : cls_ids) {
            z[domain_id][classifier_id] = topic_id;
            err_cnt[topic_id][0] += disagreement[domain_id][classifier_id];
            err_cnt[topic_id][1] += (num_example[domain_id] - disagreement[domain_id][classifier_id]);
            sum_err[topic_id] += num_example[domain_id];
        }
    }

    public int sample_one_table_collapsed(int domain_id, int table_id) {
        int cls_ids[] = update_before_sampling_tables_topic(domain_id, table_id);
        int disagree_table = 0;
        int match_table = 0;
        for (int classifier_id : cls_ids) {
            disagree_table += disagreement[domain_id][classifier_id];
            match_table += num_example[domain_id] - disagreement[domain_id][classifier_id];
        }
        int total_cnt = hdp.prob_topic_assignment_for_table(domain_id, table_id);
        double z_probabilities[] = new double[total_cnt];
        for (int i = 0; i < total_cnt; i++) {
            z_probabilities[i] = Math.log(hdp.pdf[i].prob);
        }
        for (int i = 0; i < total_cnt; i++) {
            int topic_id = hdp.pdf[i].topic;
            z_probabilities[i] += logBeta(alphae + err_cnt[topic_id][0] + disagree_table, betae + err_cnt[topic_id][1] + match_table);
            z_probabilities[i] -= logBeta(alphae + err_cnt[topic_id][0], betae + err_cnt[topic_id][1]);
        }
        for (int i = 1; i < total_cnt; i++) {
            z_probabilities[i] = sumLogProb(z_probabilities[i - 1], z_probabilities[i]);
        }
        double uniform = Math.log(rm.nextDouble()) + z_probabilities[total_cnt - 1];
        int loc = total_cnt - 1;
        for (int i = 0; i < total_cnt; i++) {
            if (z_probabilities[i] > uniform) {
                loc = i;
                break;
            }
        }
        update_after_sampling_tables_topic(domain_id, table_id, hdp.pdf[loc].topic, cls_ids);
        return hdp.pdf[loc].topic;
    }

    public int sample_one_table_uncollapsed(int domain_id, int table_id) {
        int cls_ids[] = update_before_sampling_tables_topic(domain_id, table_id);
        int disagree_table = 0;
        int match_table = 0;
        for (int classifier_id : cls_ids) {
            disagree_table += disagreement[domain_id][classifier_id];
            match_table += num_example[domain_id] - disagreement[domain_id][classifier_id];
        }
        int total_cnt = hdp.prob_topic_assignment_for_table(domain_id, table_id);
        double z_probabilities[] = new double[total_cnt];
        for (int i = 0; i < total_cnt; i++) {
            z_probabilities[i] = Math.log(hdp.pdf[i].prob);
        }
        error_rate[hdp.pdf[total_cnt - 1].topic] = (1.0 * disagree_table) / (match_table + disagree_table);
        for (int i = 0; i < total_cnt; i++) {
            int topic_id = hdp.pdf[i].topic;
            z_probabilities[i] += (disagree_table) * Math.log(error_rate[topic_id]) + (match_table) * Math.log(1 - error_rate[topic_id]);
        }
        for (int i = 1; i < total_cnt; i++) {
            z_probabilities[i] = sumLogProb(z_probabilities[i - 1], z_probabilities[i]);
        }
        double uniform = Math.log(rm.nextDouble()) + z_probabilities[total_cnt - 1];
        int loc = total_cnt - 1;
        for (int i = 0; i < total_cnt; i++) {
            if (z_probabilities[i] > uniform) {
                loc = i;
                break;
            }
        }
        update_after_sampling_tables_topic(domain_id, table_id, hdp.pdf[loc].topic, cls_ids);
        return hdp.pdf[loc].topic;
    }

    public void sample_tables_topic_collapsed() {
        for (int domain_id = 0; domain_id < num_domain; domain_id++) {
            int table_ids[] = hdp.get_tables_taken(domain_id);
            for (int table_id : table_ids) {
                sample_one_table_collapsed(domain_id, table_id);
            }
        }
    }

    public void sample_tables_topic_uncollapsed() {
        for (int domain_id = 0; domain_id < num_domain; domain_id++) {
            int table_ids[] = hdp.get_tables_taken(domain_id);
            for (int table_id : table_ids) {
                sample_one_table_uncollapsed(domain_id, table_id);
            }
        }
    }

    public void update_before_sampling_z(int domain_id, int classifier_id) {
        hdp.remove_items_table_assignment(domain_id, classifier_id);
        int topic_id = z[domain_id][classifier_id];
        err_cnt[topic_id][0] -= disagreement[domain_id][classifier_id];
        err_cnt[topic_id][1] -= (num_example[domain_id] - disagreement[domain_id][classifier_id]);
        sum_err[topic_id] -= num_example[domain_id];
    }

    public void update_after_sampling_z(int domain_id, int classifier_id, int table_id, int topic_id) {
        hdp.add_items_table_assignment(domain_id, classifier_id, table_id, topic_id);
        z[domain_id][classifier_id] = topic_id;
        err_cnt[topic_id][0] += disagreement[domain_id][classifier_id];
        err_cnt[topic_id][1] += (num_example[domain_id] - disagreement[domain_id][classifier_id]);
        sum_err[topic_id] += num_example[domain_id];
    }

    public int sample_one_z_collapsed(int domain_id, int classifier_id) {
        update_before_sampling_z(domain_id, classifier_id);
        int total_cnt = hdp.prob_table_assignment_for_item(domain_id, classifier_id);
        double z_probabilities[] = new double[total_cnt];
        for (int i = 0; i < total_cnt; i++) {
            z_probabilities[i] = Math.log(hdp.pdf[i].prob);
        }
        int topic_id = 0;
        for (int i = 0; i < total_cnt; i++) {
            topic_id = hdp.pdf[i].topic;
            z_probabilities[i] += logBeta(alphae + err_cnt[topic_id][0] + disagreement[domain_id][classifier_id], betae + err_cnt[topic_id][1] + num_example[domain_id] - disagreement[domain_id][classifier_id]);
            z_probabilities[i] -= logBeta(alphae + err_cnt[topic_id][0], betae + err_cnt[topic_id][1]);
        }
        for (int i = 1; i < total_cnt; i++) {
            z_probabilities[i] = sumLogProb(z_probabilities[i - 1], z_probabilities[i]);
        }
        double uniform = Math.log(rm.nextDouble()) + z_probabilities[total_cnt - 1];
        int loc = total_cnt - 1;
        for (int i = 0; i < total_cnt; i++) {
            if (z_probabilities[i] > uniform) {
                loc = i;
                break;
            }
        }
        update_after_sampling_z(domain_id, classifier_id, hdp.pdf[loc].table, hdp.pdf[loc].topic);
        return hdp.pdf[loc].topic;
    }

    public int sample_one_z_uncollapsed(int domain_id, int classifier_id) {
        update_before_sampling_z(domain_id, classifier_id);
        int total_cnt = hdp.prob_table_assignment_for_item(domain_id, classifier_id);
        double z_probabilities[] = new double[total_cnt];
        for (int i = 0; i < total_cnt; i++) {
            z_probabilities[i] = Math.log(hdp.pdf[i].prob);
        }
        int topic_id = 0;
        error_rate[hdp.pdf[total_cnt - 1].topic] = (1.0 * disagreement[domain_id][classifier_id]) / num_example[domain_id];
        for (int i = 0; i < total_cnt; i++) {
            topic_id = hdp.pdf[i].topic;
            z_probabilities[i] += disagreement[domain_id][classifier_id] * Math.log(error_rate[topic_id]) + (num_example[domain_id] - disagreement[domain_id][classifier_id]) * Math.log(1 - error_rate[topic_id]);
        }
        for (int i = 1; i < total_cnt; i++) {
            z_probabilities[i] = sumLogProb(z_probabilities[i - 1], z_probabilities[i]);
        }
        double uniform = Math.log(rm.nextDouble()) + z_probabilities[total_cnt - 1];
        int loc = total_cnt - 1;
        for (int i = 0; i < total_cnt; i++) {
            if (z_probabilities[i] > uniform) {
                loc = i;
                break;
            }
        }
        update_after_sampling_z(domain_id, classifier_id, hdp.pdf[loc].table, hdp.pdf[loc].topic);
        return hdp.pdf[loc].topic;
    }

    public void sample_z_collapsed() {
        for (int domain_id = 0; domain_id < num_domain; domain_id++) {
            for (int classifier_id = 0; classifier_id < num_classifier; classifier_id++) {
                sample_one_z_collapsed(domain_id, classifier_id);
            }
        }
    }

    public void sample_z_uncollapsed() {
        for (int domain_id = 0; domain_id < num_domain; domain_id++) {
            for (int classifier_id = 0; classifier_id < num_classifier; classifier_id++) {
                sample_one_z_uncollapsed(domain_id, classifier_id);
            }
        }
    }

    public void update_before_sampling_l(int domain_id, int example_id) {
        int lid = (l[domain_id][example_id] ? 1 : 0);
        int topic_id = -1;
        li_cnt[domain_id][lid]--;
        sum_li[domain_id]--;
        for (int j = 0; j < num_classifier; j++) {
            topic_id = z[domain_id][j];
            if (f[domain_id][j][example_id] != l[domain_id][example_id]) {
                err_cnt[topic_id][0]--;
            } else {
                err_cnt[topic_id][1]--;
            }
            sum_err[topic_id]--;
        }
    }

    public void update_after_sampling_l(int domain_id, int example_id, int lid) {
        l[domain_id][example_id] = (lid == 1);
        li_cnt[domain_id][lid]++;
        sum_li[domain_id]++;
        int topic_id = -1;
        for (int j = 0; j < num_classifier; j++) {
            topic_id = z[domain_id][j];
            if (f[domain_id][j][example_id] != l[domain_id][example_id]) {
                err_cnt[topic_id][0]++;
            } else {
                err_cnt[topic_id][1]++;
            }
            sum_err[topic_id]++;
        }
    }

    public int sample_one_l_colapsed(int domain_id, int example_id, TIntIntHashMap thmp) {
        update_before_sampling_l(domain_id, example_id);
        int num_topics = thmp.size();
        int cnt_err[][] = new int[num_topics][2];
        int cnt_match[][] = new int[num_topics][2];
        int topic_id = -1;
        double p0 = Math.log(betap + li_cnt[domain_id][0]);
        double p1 = Math.log(alphap + li_cnt[domain_id][1]);
        for (int j = 0; j < num_classifier; j++) {
            topic_id = z[domain_id][j];
            if (f[domain_id][j][example_id]) {
                p0 += Math.log(alphae + err_cnt[topic_id][0] + cnt_err[thmp.get(topic_id)][0]);
                p1 += Math.log(betae + err_cnt[topic_id][1] + cnt_match[thmp.get(topic_id)][1]);
                cnt_err[thmp.get(topic_id)][0]++;
                cnt_match[thmp.get(topic_id)][1]++;
            } else {
                p0 += Math.log(betae + err_cnt[topic_id][1] + cnt_match[thmp.get(topic_id)][0]);
                p1 += Math.log(alphae + err_cnt[topic_id][0] + cnt_err[thmp.get(topic_id)][1]);
                cnt_match[thmp.get(topic_id)][0]++;
                cnt_err[thmp.get(topic_id)][1]++;
            }
        }
        double sum = sumLogProb(p0, p1);
        double val = Math.log(rm.nextDouble()) + sum;
        int lid = 1;
        if (val < p0) {
            lid = 0;
        }
        update_after_sampling_l(domain_id, example_id, lid);
        return lid;
    }

    public int sample_one_l_uncolapsed(int domain_id, int example_id, TIntIntHashMap thmp) {
        update_before_sampling_l(domain_id, example_id);
        int num_topics = thmp.size();
        int cnt_err[][] = new int[num_topics][2];
        int cnt_match[][] = new int[num_topics][2];
        int topic_id = -1;
        double p0 = Math.log(betap + li_cnt[domain_id][0]);
        double p1 = Math.log(alphap + li_cnt[domain_id][1]);
        for (int j = 0; j < num_classifier; j++) {
            topic_id = z[domain_id][j];
            if (f[domain_id][j][example_id]) {
                p0 += Math.log(error_rate[topic_id]);
                p1 += Math.log(1 - error_rate[topic_id]);
            } else {
                p0 += Math.log(1 - error_rate[topic_id]);
                p1 += Math.log(error_rate[topic_id]);
            }
        }
        double sum = sumLogProb(p0, p1);
        double val = Math.log(rm.nextDouble()) + sum;
        int lid = 1;
        if (val < p0) {
            lid = 0;
        }
        update_after_sampling_l(domain_id, example_id, lid);
        return lid;
    }

    public void sample_l_colapsed() {
        int topics[] = hdp.get_topics();
        TIntIntHashMap thmp = new TIntIntHashMap();
        for (int i = 0; i < topics.length; i++) {
            thmp.put(topics[i], i);
        }
        for (int domain_id = 0; domain_id < num_domain; domain_id++) {
            for (int example_id = 0; example_id < num_example[domain_id]; example_id++) {
                sample_one_l_colapsed(domain_id, example_id, thmp);
            }
        }
        disagreement = new int[num_domain][num_classifier];
        for (int domain_id = 0; domain_id < num_domain; domain_id++) {
            for (int classifier_id = 0; classifier_id < num_classifier; classifier_id++) {
                for (int example_id = 0; example_id < num_example[domain_id]; example_id++) {
                    disagreement[domain_id][classifier_id] += (f[domain_id][classifier_id][example_id] == l[domain_id][example_id]) ? 0 : 1;
                }
            }
        }
    }

    public void sample_l_uncolapsed() {
        int topics[] = hdp.get_topics();
        TIntIntHashMap thmp = new TIntIntHashMap();
        for (int i = 0; i < topics.length; i++) {
            thmp.put(topics[i], i);
        }
        for (int domain_id = 0; domain_id < num_domain; domain_id++) {
            for (int example_id = 0; example_id < num_example[domain_id]; example_id++) {
                sample_one_l_uncolapsed(domain_id, example_id, thmp);
            }
        }
        disagreement = new int[num_domain][num_classifier];
        for (int domain_id = 0; domain_id < num_domain; domain_id++) {
            for (int classifier_id = 0; classifier_id < num_classifier; classifier_id++) {
                for (int example_id = 0; example_id < num_example[domain_id]; example_id++) {
                    disagreement[domain_id][classifier_id] += (f[domain_id][classifier_id][example_id] == l[domain_id][example_id]) ? 0 : 1;
                }
            }
        }
    }

    public void gibbs_one_iteration_collapsed() {
        sample_z_collapsed();
        check_error_negative();
        sample_tables_topic_collapsed();
        check_error_negative();
        sample_l_colapsed();
        check_error_negative();
    }

    public void run_gibbs_collapsed(int num_iteration) {
        for (int i = 0; i < num_iteration; i++) {
            gibbs_one_iteration_collapsed();
            
        }
    }
    
    public void check_error_negative(){
//        for(int i=0;i<err_cnt.length;i++){
//            if(err_cnt[i][0] < 0 || err_cnt[i][1] < 0){
//                System.out.println("Error");
//            }
//        }
    }

    public void gibbs_one_iteration_uncollapsed() {
        sample_error_rate_uncollapsed();
        check_error_negative();
        sample_z_uncollapsed();
        check_error_negative();
        sample_tables_topic_uncollapsed();
        check_error_negative();
        sample_l_uncolapsed();
        check_error_negative();
    }
    
    public double rates_to_return[][];
    public double labels_to_return[][];
    public int num_cluster;
    public double avg_error_rates[];
    public void run_gibbs_uncollapsed(int num_burn_in, int num_internal_iteration, int num_samples_needed) {
        avg_error_rates = new double[error_rate.length];
        rates_to_return = new double[num_domain][num_classifier];
        labels_to_return = new double[num_domain][];
        for(int d=0;d<num_domain;d++){
            labels_to_return[d] = new double[num_example[d]];
        }
        for (int i = 0; i < num_burn_in; i++) {
            gibbs_one_iteration_uncollapsed();
        }
        for (int k = 0; k < num_samples_needed; k++) {
            for (int i = 0; i < num_internal_iteration; i++) {
                gibbs_one_iteration_uncollapsed();
            }
            for(int d=0;d<num_domain;d++){
                for(int j=0;j<num_classifier;j++){
                    rates_to_return[d][j] += (error_rate[z[d][j]]/num_samples_needed);                    
                }
                for(int i=0;i<num_example[d];i++){
                    labels_to_return[d][i] += l[d][i]? (1.0/num_samples_needed):0;
                }
            }
            for(int p=0;p<error_rate.length;p++){
                avg_error_rates[p] += error_rate[p]/num_samples_needed;
            }
        }
        num_cluster = hdp.get_topics().length;
        
    }
    
    public double get_log_likelihood(List<boolean [][]> fun, double alpha, double gamma, int num_iteration, int licnt[][]){
        ErrorEstimationDomainsHDPNew hdp1 = new ErrorEstimationDomainsHDPNew(fun, 1e-100, 1, avg_error_rates,licnt);
        for(int i=0;i<num_iteration;i++){
            hdp1.sample_for_likelihood();
        }
        double log_prob =0;
        for(int it=0;it<num_iteration;it++) {
            hdp1.sample_for_likelihood();
            int zs[][] = hdp1.get_z();
            boolean ls[][] = hdp1.get_l();
            for (int d = 0; d < fun.size(); d++) {
                for (int i = 0; i < fun.get(d).length; i++) {
                    double a = (1.0 * licnt[d][0] + 1) / (licnt[d][0] + licnt[d][1] + 2);
                    a = ls[d][i] ? a : (1 - a);
                    log_prob += Math.log(a);
                    for (int j = 0; j < fun.get(d)[i].length; j++) {
                        if (fun.get(d)[i][j] != ls[d][i]) {
                            if (!Double.isInfinite(Math.log(avg_error_rates[zs[d][j]]))) {
                                log_prob += Math.log(avg_error_rates[zs[d][j]]);
                            } else {
                                log_prob += Math.log(alphae / (alphae + betae));
                            }

                        } else {
                            if (!Double.isInfinite(Math.log(1 - avg_error_rates[zs[d][j]]))) {
                                log_prob += Math.log(1 - avg_error_rates[zs[d][j]]);
                            } else {
                                log_prob += Math.log(betae / (alphae + betae));
                            }
                        }
                    }
                }
            }
        }
        return log_prob / num_iteration;
    }
    
    public void sample_for_likelihood(){
        sample_z_uncollapsed();
        check_error_negative();
        sample_tables_topic_uncollapsed();
        check_error_negative();
        sample_l_uncolapsed();
        check_error_negative();
    }

    public void sample_error_rate_uncollapsed() {
        int topics[] = hdp.get_topics();
        for (int topic_id : topics) {
//            error_rate[topic_id] = randomDataGenerator.nextBeta(alphae + err_cnt[topic_id][0], betae + err_cnt[topic_id][1]);
            error_rate[topic_id] = (alphae + err_cnt[topic_id][0]) / (alphae + err_cnt[topic_id][0] + betae + err_cnt[topic_id][1]);
        }
    }
    
    public int[][] get_z(){
        return z;
    }
    
    public boolean[][] get_l(){
        return l;
    }

    public static double sumLogProb(double a, double b) {
        if (a == Double.NEGATIVE_INFINITY) {
            return b;
        } else if (b == Double.NEGATIVE_INFINITY) {
            return a;
        } else if (b < a) {
            return a + Math.log(1 + Math.exp(b - a));
        } else {
            return b + Math.log(1 + Math.exp(a - b));
        }
    }

    public static void main(String argv[]) {
        int num_classifier = 3;
        int num_domain = 2;
        int num_example[] = {4, 4};
        boolean f[][][] = new boolean[2][3][4];
        Random rm = new Random();
        for (int p = 0; p < num_domain; p++) {
            for (int i = 0; i < num_example[p]; i++) {
                double prob = rm.nextDouble();
                for (int j = 0; j < num_classifier; j++) {
                    f[p][j][i] = (rm.nextDouble() < prob);
                }
            }
        }

        double alphap = 1.0;
        double betap = 100;
        double alphae = 1;
        double betae = 100;

//        HDPErrorRate(boolean f[][][], double alphap, double betap, double alphae, double betae, int num_example[],
//            int num_domain, int num_classifier, double alpha, double gamma)
//        HDPErrorRate hp = new HDPErrorRate(f, alphap,
//                betap, alphae, betae, num_example, num_domain, num_classifier, 1, 1);
//        hp.gibbs_one_iteration();
    }

}
