package cs446.homework2;

import java.io.File;
import java.io.FileReader;
import java.util.*;

import weka.classifiers.*;
import weka.core.*;
import weka.core.converters.ArffSaver;
import cs446.weka.classifiers.trees.Id3;

public class StumpTester {

    private static HashMap<String,Attribute> attrLookup = new HashMap<String,Attribute>();
    private static FastVector zeroOne;
    private static FastVector labels;

    private static Instance makeInstance(double[] features, double value, Instances instances) {
	Instance instance = new Instance(101);
	instance.setDataset(instances);
	for(int i = 0; i < 100; i++) {
	    instance.setValue(attrLookup.get("stump"+i), (int)(features[i])+"");
	}
	instance.setClassValue((value == 0.0 ? "-" : "+"));
	return instance;
    }

    public static void main(String[] args) throws Exception {

	if (args.length != 2) {
	    System.err.println("Usage: 1 full set, 1 path to save arff");
	    System.exit(-1);
	}

	// Load the data
	Instances data = new Instances(new FileReader(new File(args[0])));

	// The last attribute is the class label
	data.setClassIndex(data.numAttributes() - 1);

	Id3[] stumps = new Id3[100];
	
	for(int i = 0; i < 100; i++) { 
	    //make stump
	    data.randomize(new java.util.Random());
	    Instances stumpData = data.trainCV(2,0);
	    stumps[i] = new Id3();
	    stumps[i].setMaxDepth(-1);
	    stumps[i].buildClassifier(stumpData);	    
	}

	zeroOne = new FastVector(2);
	zeroOne.addElement("1");
	zeroOne.addElement("0");

	labels = new FastVector(2);
	labels.addElement("+");
	labels.addElement("-");

	FastVector attributes = new FastVector();
	for(int i = 0; i < 100; i++) {
	    Attribute newAttr = new Attribute("stump"+i, zeroOne);
	    attrLookup.put("stump"+i,newAttr);
	    attributes.addElement(newAttr);
	}
	
	Attribute classLabel = new Attribute("Class", labels);
	attrLookup.put("Class", classLabel);
	attributes.addElement(classLabel);

	// make instances
	Instances richFeatures = new Instances("Decision Stumps", attributes, 0);
	richFeatures.setClass(classLabel);
	for(int i = 0; i < data.numInstances(); i++) {
	    Instance cur = data.instance(i);
	    double[] predictions = new double[100];
	    for(int j = 0; j < 100; j++){
		predictions[j] = stumps[j].classifyInstance(cur);
	    }
	    richFeatures.add(makeInstance(predictions, cur.classValue(), richFeatures));
	}

	ArffSaver saver = new ArffSaver();
	saver.setInstances(richFeatures);
	saver.setFile(new File(args[1]));
	saver.writeBatch();

	double numCorrect = 0;
	double numTotal = 0;

	// five-fold cross validation
	for(int i=0; i<5; i++){
	    Instances train = richFeatures.trainCV(5,i);
	    Instances test = richFeatures.testCV(5,i);

	    SGD classifier = new SGD();
	    classifier.buildClassifier(train);

	    Evaluation evaluation = new Evaluation(test);
	    evaluation.evaluateModel(classifier, test);
	    System.out.println(evaluation.toSummaryString());

	    numCorrect += evaluation.correct();
	    numTotal += evaluation.correct() + evaluation.incorrect();	
	}

	System.out.println("SGD(stump) numCorrect percentage across cv is: " + numCorrect / numTotal * 100 + " %");
		    

    }
}
