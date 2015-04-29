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
String dbpath = config.getString("dbpath", defaultPath + File.separator + "uci-trust-test")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, './uci-trust', true), config)

def dir = 'data'+java.io.File.separator+'ucidata'+java.io.File.separator;

/*
 * Now we can initialize a PSLModel, which is the core component of PSL.
 * The first constructor argument is the context in which the PSLModel is defined.
 * The second argument is the DataStore we will be using.
 */

PSLModel m = new PSLModel(this, data)

// we care about 'trusts', but add in 'knows', so that we know which pairs to consider
m.add predicate: "trusts", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "knows" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

// add network style triads
//m.add rule: ( trusts(A,B) & trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C) & (A^B) & (A^C) & (B^C)) >> trusts(A,C), weight: 1, squared:false

m.add rule: ( trusts(A,B) & trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C) & (A^B) & (A^C) & (B^C)) >> trusts(A,C), weight: 1
m.add rule: ( trusts(A,B) & ~trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C) & (A^B) & (A^C) & (B^C)) >> ~trusts(A,C), weight: 1
m.add rule: ( ~trusts(A,B) & trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C) & (A^B) & (A^C) & (B^C)) >> ~trusts(A,C), weight: 1
m.add rule: ( ~trusts(A,B) & ~trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C) & (A^B) & (A^C) & (B^C)) >> trusts(A,C), weight: 1

m.add rule: ( trusts(A,B) & trusts(A,C) & knows(A,B) & knows(A,C) & knows(B,C) & (A^B) & (A^C) & (B^C)) >> trusts(B,C), weight: 1
m.add rule: ( trusts(A,B) & ~trusts(A,C) & knows(A,B) & knows(A,C) & knows(B,C) & (A^B) & (A^C) & (B^C)) >> ~trusts(B,C), weight: 1
m.add rule: ( ~trusts(A,B) & trusts(A,C) & knows(A,B) & knows(A,C) & knows(B,C) & (A^B) & (A^C) & (B^C)) >> ~trusts(B,C), weight: 1
m.add rule: ( ~trusts(A,B) & ~trusts(A,C) & knows(A,B) & knows(A,C) & knows(B,C) & (A^B) & (A^C) & (B^C)) >> trusts(B,C), weight: 1

m.add rule: ( trusts(A,C) & trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C) & (A^B) & (A^C) & (B^C)) >> trusts(A,B), weight: 1
m.add rule: ( trusts(A,C) & ~trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C) & (A^B) & (A^C) & (B^C)) >> ~trusts(A,B), weight: 1
m.add rule: ( ~trusts(A,C) & trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C) & (A^B) & (A^C) & (B^C)) >> ~trusts(A,B), weight: 1
m.add rule: ( ~trusts(A,C) & ~trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C) & (A^B) & (A^C) & (B^C)) >> trusts(A,B), weight: 1


/*
m.add rule: ( trusts(A,B) & trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> trusts(A,C), weight: 1
m.add rule: ( trusts(A,B) & ~trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> ~trusts(A,C), weight: 1
m.add rule: ( ~trusts(A,B) & trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> ~trusts(A,C), weight: 1
m.add rule: ( ~trusts(A,B) & ~trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> trusts(A,C), weight: 1

m.add rule: ( trusts(A,B) & trusts(A,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> trusts(B,C), weight: 1
m.add rule: ( trusts(A,B) & ~trusts(A,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> ~trusts(B,C), weight: 1
m.add rule: ( ~trusts(A,B) & trusts(A,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> ~trusts(B,C), weight: 1
m.add rule: ( ~trusts(A,B) & ~trusts(A,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> trusts(B,C), weight: 1

m.add rule: ( trusts(A,C) & trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> trusts(A,B), weight: 1
m.add rule: ( trusts(A,C) & ~trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> ~trusts(A,B), weight: 1
m.add rule: ( ~trusts(A,C) & trusts(B,C) & knows(A,B) & knows(A,C) & knows(B,C)) >> ~trusts(A,B), weight: 1
m.add rule: ( ~(trusts(A,C) | trusts(B,C)) & knows(A,B) & knows(A,C) & knows(B,C)) >> trusts(A,B), weight: 1
*/

// this might be cheating: basically allows us to learn a prior over distribution of +1/-1
m.add rule: (knows(A,B)) >> trusts(A,B), weight: 0.1
m.add rule: (knows(A,B)) >> ~trusts(A,B), weight: 0.1

// add in a constraint so that there's no trust between ppl that don't know each other
// actually, don't need it for the non-loopy case
//m.add rule: ~knows(A,B) >> ~trusts(A,B), constraint: true

println m;

// define a partition with known truth values (X)
def fixed_partition = new Partition(0);

// add in the data about who knows who
knows_inserter = data.getInserter(knows, fixed_partition)
InserterUtils.loadDelimitedData(knows_inserter, dir+"knows.txt");

// add in the training data about who trusts who
def trusts_inserter = data.getInserter(trusts, fixed_partition);
InserterUtils.loadDelimitedDataTruth(trusts_inserter, dir+"train.txt");

// make a write partition (Y) so that we can lock the above results as a fixed partition
def write_partition = new Partition(1);


// Create a database from these partitions
Database db = data.getDatabase(write_partition, fixed_partition);

println "Training data:"
for (GroundAtom atom : Queries.getAllAtoms(db, Trusts))
	println atom.toString() + "\t" + atom.getValue();

// run inference with the given weights
LazyMPEInference inferenceApp = new LazyMPEInference(m, db, config);
inferenceApp.mpeInference();
inferenceApp.close();

// Display the results
println "Inference results with hand-defined weights:"
for (GroundAtom atom : Queries.getAllAtoms(db, Trusts))
	println atom.toString() + "\t" + atom.getValue();


// Create a new partition to do weight-learning
Partition trueDataPartition = new Partition(2);

// insert the true values of trust for the test data
trueInserter = data.getInserter(trusts, trueDataPartition)
InserterUtils.loadDelimitedDataTruth(trueInserter, dir + "test.txt");

// make a new database with just the true partition (test data) and everything fixed
Database trueDataDB = data.getDatabase(trueDataPartition, [Knows, Trusts] as Set);

println "Test data:"
for (GroundAtom atom : Queries.getAllAtoms(trueDataDB, Trusts))
	println atom.toString() + "\t" + atom.getValue();


// do weight learning based on the test data
LazyMaxLikelihoodMPE weightLearning = new LazyMaxLikelihoodMPE(m, db, trueDataDB, config);
weightLearning.learn();
weightLearning.close();

// Display the new model
println "Learned model:"
println m

// Apply new model, to see if we've improved
LazyMPEInference inferenceApp2 = new LazyMPEInference(m, db, config);
inferenceApp2.mpeInference();
inferenceApp2.close();

// Display results
println "Inference results with learned weights:"
for (GroundAtom atom : Queries.getAllAtoms(db, Trusts))
	println atom.toString() + "\t" + atom.getValue();




