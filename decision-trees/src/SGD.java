package cs446.homework2;

import weka.core.*;
import weka.classifiers.*;

import java.util.*;
import java.lang.Exception;
import java.lang.Math;

public class SGD extends Classifier {
    //learning rate
    private double alpha = 0.0001;
    //normal vector
    private double[] w;
    //threshold
    private double error_threshold = 80.0;

    @Override
    public void buildClassifier(Instances data) throws Exception {
	// The last attribute is the class label
	int numFeatures = data.numAttributes() - 1;
	//initial normal vector
	w = new double[numFeatures];
	for(int i=0; i<numFeatures; i++){
	    w[i] = 1.0;
	}

	double error = sgd_it(data);

	int numIter = 0;
	while(error > error_threshold){
	    double curr_error = sgd_it(data);
	    error = Math.abs(error - curr_error);
	    //System.out.println(error);
	    numIter++;
	}
	/*
	System.out.println("----w----");
	for(int i=0; i<w.length; i++){
	    System.out.print(w[i] + " ");
	}
	System.out.println();
	*/
	System.out.println("Count: "+numIter);

    }

    /**
     * single iteration of all instances
     */
    private double sgd_it(Instances data)throws Exception{
	//shuffle instances
	data.randomize(new Random());
	/*
	double[] x_sum = new double[data.numAttributes() - 1];
	for(int i=0; i<x_sum.length; i++){
	    x_sum[i] = 0.0;
	}
	*/
	double error = 0.0;
	Enumeration instEnum = data.enumerateInstances();
	while(instEnum.hasMoreElements()){
	    Instance inst = (Instance)instEnum.nextElement();
	    //current instance as vector
	    double[] x = removeLast(inst.toDoubleArray());
	    //y=1 or y=-1
	    double y = inst.classValue() == 1.0 ? 1.0 : -1.0;
	    //LMS(Adaline)
	    double gradient = y - dot(w,x);
	    error += 0.5 * Math.pow(gradient, 2);


	    //x_sum = addVectors(x_sum, multiplyScalar(gradient, x));


	    //update w
	    for(int j=0; j<w.length; j++){
	    	w[j] = w[j] + alpha * gradient * x[j];
	    }
	}
	//int m = data.numInstances();
	//double[] old_w = multiplyScalar(1.0, w);
	//w = addVectors(w, multiplyScalar(alpha/m, x_sum));
	/*
	System.out.println("----NUMINSTANCES----");
	System.out.println(1.0/m);

	System.out.println("----SCALAR----");
	for(int i=0; i<x_sum.length; i++){
	    System.out.print(multiplyScalar(1/m, x_sum)[i] + " ");
	}
	System.out.println();

	System.out.println("----X_SUM----");
	for(int i=0; i<x_sum.length; i++){
	    System.out.print(x_sum[i] + " ");
	}
	System.out.println();

	System.out.println("----w----");
	for(int i=0; i<w.length; i++){
	    System.out.print(w[i] + " ");
	}
	System.out.println();
	*/

	return error;
    }

    @Override
    public double classifyInstance(Instance instance)throws Exception{
	double[] x = removeLast(instance.toDoubleArray());
	double label = (dot(w,x) >= 0.0) ? 1.0 : 0.0;
	return label;
    }

    /**
     * dot product of two vectors
     */
    private double dot(double[] v1, double[] v2)throws Exception{
	if(v1.length != v2.length){
	    throw new Exception("Vector sizes mismatch.");
	}
	double res = 0.0;
	for(int i=0; i<v1.length; i++){
	    res += v1[i]*v2[i];
	}
	return res;
    }

    private double[] multiplyScalar(double scalar, double[] vector){
	double[] res = new double[vector.length];
	for(int i=0; i<res.length; i++){
	    res[i] = scalar * vector[i];
	}
	return res;
    }

    private double[] addVectors(double[] v1, double[] v2)throws Exception{
	if(v1.length != v2.length){
	    throw new Exception("Vector sizes mismatch.");
	}
	double[] res = new double[v1.length];
	for(int i=0; i<v1.length; i++){
	    res[i] = v1[i] + v2[i];
	}
	return res;
    }

    /**
     * remove the last element in an array, also adjusts array's size
     */
    private double[] removeLast(double[] array){
	double[] res = new double[array.length - 1];
	for(int i=0; i<res.length; i++){
	    res[i] = array[i];
	}
	return res;
    }

    /**
     * main function
     */	
    public static void main(String[] args) {
	System.out.println("SGD");
    }


}

