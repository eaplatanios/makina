/*
 * This file is part of the PSL software.
 * Copyright 2011-2013 University of Maryland
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.umd.cs.example;

import edu.umd.cs.psl.application.inference.LazyMPEInference;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.LazyMaxLikelihoodMPE;
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.Partition;
import edu.umd.cs.psl.database.ReadOnlyDatabase;
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.PredicateConstraint;
import edu.umd.cs.psl.groovy.SetComparison;
import edu.umd.cs.psl.model.argument.ArgumentType;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.function.ExternalFunction;
import edu.umd.cs.psl.ui.functions.textsimilarity.*
import edu.umd.cs.psl.ui.loading.InserterUtils;
import edu.umd.cs.psl.util.database.Queries;

// set things up
ConfigManager cm = ConfigManager.getManager()
ConfigBundle config = cm.getBundle("uci-trust-test")

def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = config.getString("dbpath", defaultPath + File.separator + "epinions")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, './epinions', true), config)

def data_dir = 'data'+java.io.File.separator;
def output_dir = 'output'+java.io.File.separator;

/*
 * Now we can initialize a PSLModel, which is the core component of PSL.
 * The first constructor argument is the context in which the PSLModel is defined.
 * The second argument is the DataStore we will be using.
 */
PSLModel m = new PSLModel(this, data)

println "Creating model"

// we care about 'trusts', but add in 'knows', so that we know which pairs to consider
m.add predicate: "trusts", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "knows" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

// add network style triads
//m.add rule: ( trusts(A,B) & trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C) & (A^B) & (A^C) & (B^C)) >> trusts(A,C), weight: 1, squared:false
m.add rule: ( trusts(A,B) & trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> trusts(A,C), weight: 1
m.add rule: ( trusts(A,B) & ~trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> ~trusts(A,C), weight: 1
m.add rule: ( ~trusts(A,B) & trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> ~trusts(A,C), weight: 1
m.add rule: ( ~trusts(A,B) & ~trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> trusts(A,C), weight: 1

m.add rule: ( trusts(A,B) & trusts(C,B) & knows(A,B) & knows(A,C) & knows(B,C)) >> trusts(A,C), weight: 1
m.add rule: ( trusts(A,B) & ~trusts(C,B) & knows(A,B) & knows(A,C) & knows(B,C)) >> ~trusts(A,C), weight: 1
m.add rule: ( ~trusts(A,B) & trusts(C,B) & knows(A,B) & knows(A,C) & knows(B,C)) >> ~trusts(A,C), weight: 1
m.add rule: ( ~trusts(A,B) & ~trusts(C,B) & knows(A,B) & knows(A,C) & knows(B,C)) >> trusts(A,C), weight: 1

m.add rule: ( trusts(B,A) & trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> trusts(A,C), weight: 1
m.add rule: ( trusts(B,A) & ~trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> ~trusts(A,C), weight: 1
m.add rule: ( ~trusts(B,A) & trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> ~trusts(A,C), weight: 1
m.add rule: ( ~trusts(B,A) & ~trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> trusts(A,C), weight: 1

m.add rule: ( trusts(B,A) & trusts(C,B) & knows(A,B) & knows(A,C) & knows(B,C)) >> trusts(A,C), weight: 1
m.add rule: ( trusts(B,A) & ~trusts(C,B) & knows(A,B) & knows(A,C) & knows(B,C)) >> ~trusts(A,C), weight: 1
m.add rule: ( ~trusts(B,A) & trusts(C,B) & knows(A,B) & knows(A,C) & knows(B,C)) >> ~trusts(A,C), weight: 1
m.add rule: ( ~trusts(B,A) & ~trusts(C,B) & knows(A,B) & knows(A,C) & knows(B,C)) >> trusts(A,C), weight: 1


// this might be cheating: basically allows us to learn a prior over distribution of +1/-1
//m.add rule: (knows(A,B)) >> trusts(A,B) | trusts(A,B), constraint: true
m.add rule: (knows(A,B)) >> trusts(A,B), weight: 0.1
m.add rule: (knows(A,B)) >> ~trusts(A,B), weight: 0.1

// add in a constraint so that there's no trust between ppl that don't know each other
// actually, don't need it for the non-loopy case
m.add rule: ~knows(A,B) >> ~trusts(A,B), constraint: true

// write the model to file
new File(output_dir+"model_before_learning.txt").withWriter { out ->
	out.println m;
}


// Load fixed data about who knows who (all) and who trusts who (some)
println "Loading fixed data"

// define a partition with known truth values (X)
def fixed_partition = new Partition(0);

// add in the data about who knows who (presence = 1)
knows_inserter = data.getInserter(knows, fixed_partition)
InserterUtils.loadDelimitedData(knows_inserter, data_dir+"knows.txt");

// add in the training data about who trusts who
def trusts_inserter = data.getInserter(trusts, fixed_partition);
InserterUtils.loadDelimitedDataTruth(trusts_inserter, data_dir+"train.txt");

// make a write partition (Y) so that we can lock the above results as a fixed partition
def write_partition = new Partition(1);

// Create a database from these partitions
Database db = data.getDatabase(write_partition, fixed_partition);

// Output the training data as a check
new File(output_dir+"training_data.txt").withWriter { out ->
	for (GroundAtom atom : Queries.getAllAtoms(db, Trusts))
		out.println atom.toString() + "\t" + atom.getValue();
}



// run inference with the manually-defined weights
println "Running inference"
LazyMPEInference inferenceApp = new LazyMPEInference(m, db, config);
inferenceApp.mpeInference();
inferenceApp.close();

/*
// Write the learned predicted truth-values to file
new File(output_dir+"output_before_learning.txt").withWriter { out ->
	for (GroundAtom atom : Queries.getAllAtoms(db, Trusts))
		out.println atom.toString() + "\t" + atom.getValue();
}


println "Loading dev data"
// Create a new partition to do weight-learning
Partition dev_partition = new Partition(2);

// insert the true values of trust for the test data
dev_inserter = data.getInserter(trusts, dev_partition)
InserterUtils.loadDelimitedDataTruth(dev_inserter, data_dir + "dev.txt");

// make a new database with just the true partition (test data) and everything fixed
Database dev_db = data.getDatabase(dev_partition, [Knows] as Set);

// Write the dev data to file as a check
new File(output_dir+"dev_data.txt").withWriter { out ->
	for (GroundAtom atom : Queries.getAllAtoms(dev_db, Trusts))
		out.println atom.toString() + "\t" + atom.getValue();
}

// do weight learning based on the test data
println "Learning weights..."
LazyMaxLikelihoodMPE weightLearning = new LazyMaxLikelihoodMPE(m, db, dev_db, config);
weightLearning.learn();
weightLearning.close();

new File(output_dir+"model_after_learning.txt").withWriter { out ->
	out.println m
}


// Apply new model, to see if we've improved
println "Running inference"
LazyMPEInference inferenceApp2 = new LazyMPEInference(m, db, config);
inferenceApp2.mpeInference();
inferenceApp2.close();

// Display results
new File(output_dir+"output_after_learning.txt").withWriter { out ->
	for (GroundAtom atom : Queries.getAllAtoms(db, Trusts))
		out.println atom.toString() + "\t" + atom.getValue();
}

*/