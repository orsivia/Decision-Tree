package cs446.homework2;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class FeatureGenerator {

    static String[] features;
    private static FastVector zeroOne;
    private static FastVector labels;

    static {
	features = new String[] { "firstName0", "firstNameN", "lastName0", "lastNameN" };

	List<String> ff = new ArrayList<String>();

	//first and last letters of firstname and lastname
	for (String f : features) {
	    for (char letter = 'a'; letter <= 'z'; letter++) {
		ff.add(f + "=" + letter);
	    }
	}

	//length of firstname and lastname
	String[] numericalFeatures = new String[]{"firstLength", "lastLength"};
	for(String f : numericalFeatures){
	    for(int i=1; i<=15; i++){
		ff.add(f + "=" + i);
	    }
	}

	features = ff.toArray(new String[ff.size()]);

	zeroOne = new FastVector(2);
	zeroOne.addElement("1");
	zeroOne.addElement("0");

	labels = new FastVector(2);
	labels.addElement("+");
	labels.addElement("-");
    }

    public static Instances readData(String fileName) throws Exception {

	Instances instances = initializeAttributes();
	Scanner scanner = new Scanner(new File(fileName));

	while (scanner.hasNextLine()) {
	    String line = scanner.nextLine();

	    Instance instance = makeInstance(instances, line);

	    instances.add(instance);
	}

	scanner.close();

	return instances;
    }

    private static Instances initializeAttributes() {

	String nameOfDataset = "Badges";

	Instances instances;

	FastVector attributes = new FastVector(9);
	for (String featureName : features) {
	    attributes.addElement(new Attribute(featureName, zeroOne));
	}
	Attribute classLabel = new Attribute("Class", labels);
	attributes.addElement(classLabel);

	instances = new Instances(nameOfDataset, attributes, 0);

	instances.setClass(classLabel);

	return instances;

    }

    private static Instance makeInstance(Instances instances, String inputLine) {
	inputLine = inputLine.trim();

	String[] parts = inputLine.split("\\s+");
	String label = parts[0];
	String firstName = parts[1].toLowerCase();
	String lastName = parts[2].toLowerCase();

	Instance instance = new Instance(features.length + 1);
	instance.setDataset(instances);

	Set<String> feats = new HashSet<String>();

	feats.add("firstName0=" + firstName.charAt(0));
	feats.add("firstNameN=" + firstName.charAt(firstName.length() - 1));
	feats.add("lastName0=" + lastName.charAt(0));
	feats.add("lastNameN=" + lastName.charAt(lastName.length() - 1));

	for (int featureId = 0; featureId < features.length; featureId++) {
	    Attribute att = instances.attribute(features[featureId]);

	    String name = att.name();
	    String featureLabel;
	    if(name.contains("Length")){
		if(name.contains("first")){
		    String[] sub = name.split("=");
		    String lengthString = sub[1];
		    if(firstName.length() == Integer.parseInt(lengthString))
			featureLabel = "1";
		    else
			featureLabel = "0";
		}
		else{
		    String[] sub = name.split("=");
		    String lengthString = sub[1];
		    if(lastName.length() == Integer.parseInt(lengthString))
			featureLabel = "1";
		    else
			featureLabel = "0";
		}
	    }
	    else{
		if (feats.contains(name)) {
		    featureLabel = "1";
		} else
		    featureLabel = "0";
	    }
	    instance.setValue(att, featureLabel);
	}

	instance.setClassValue(label);

	return instance;
    }

    public static void main(String[] args) throws Exception {

	if (args.length != 2) {
	    System.err
		    .println("Usage: FeatureGenerator input-badges-file features-file");
	    System.exit(-1);
	}
	Instances data = readData(args[0]);

	ArffSaver saver = new ArffSaver();
	saver.setInstances(data);
	saver.setFile(new File(args[1]));
	saver.writeBatch();
    }
}
