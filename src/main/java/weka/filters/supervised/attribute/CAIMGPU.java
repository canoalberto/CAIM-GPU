package weka.filters.supervised.attribute;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import weka.core.Attribute;
import weka.core.Instances;

public class CAIMGPU extends CAIMCPU
{
	private static final long serialVersionUID = 1L;
	
	private Instances Data;
	
	public native void initializeGPU(CAIMGPU algorithm, int attribute, int numberClasses, int numberAttributes, int numberInstances);
	
	public boolean batchFinished()  throws Exception
	{
		Data = getInputFormat();
		SchemeList = new ArrayList<>();
		Attribute ClassAttribute = Data.attribute(Data.numAttributes()-1);
		
		Enumeration<Object> enu = ClassAttribute.enumerateValues();
		ClassValueList = new ArrayList<String>();
		
		while(enu.hasMoreElements())
			ClassValueList.add((String)enu.nextElement());
		
		try {
			System.loadLibrary("gpu");
		} catch (Exception e) {
			throw new Exception("Can't load the GPU library. Please make sure to include gpu library path");
		}
		
		for (int current=0; current<Data.numAttributes(); current++)
			SchemeList.add(new ArrayList<Double>());
		
		ExecutorService threadExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
		
		for (int current=0;current<Data.numAttributes()-1;current++)
		{
			if (!Data.attribute(current).isNumeric()) continue;
			if (!m_DiscretizeCols.isInRange(current)) continue;
			
			threadExecutor.execute(new evaluationThread(this, current));
		}
		
		threadExecutor.shutdown();
		
		try
		{
			if (!threadExecutor.awaitTermination(30, TimeUnit.DAYS))
				System.out.println("Threadpool timeout occurred");
		}
		catch (InterruptedException ie)
		{
			System.out.println("Threadpool prematurely terminated due to interruption in thread that created pool");
		}

		if (getInputFormat() == null)
			throw new IllegalStateException("No input instance format defined");
		
		setOutputFormat();
		
		for(int i = 0; i < getInputFormat().numInstances(); i++)
			convertInstance(getInputFormat().instance(i));
		
		return true;
	}

	public float[] getAttributeValues(int attribute)
	{
		float[] attributeValues = new float[Data.numInstances()];
		
		for (int i = 0; i < Data.numInstances(); i++)
			attributeValues[i] = (float) Data.instance(i).value(attribute);
		
		return attributeValues;
	}
	
	public int[] getClassValues()
	{
		int[] classValues = new int[Data.numInstances()];
		
		for (int i = 0; i < Data.numInstances(); i++)
			classValues[i] = (int) Data.instance(i).classValue();
		
		return classValues;
	}
	
	public void addInterval(int attribute, float intervalValue)
	{
		SchemeList.get(attribute).add(1.0*intervalValue);
	}
	
	/////////////////////////////////////////////////////////////////
	// -------------------------------------------- Evaluation Thread
	/////////////////////////////////////////////////////////////////

	private class evaluationThread extends Thread
	{
		private int attribute;
		private CAIMGPU algorithm;

		public evaluationThread(CAIMGPU algorithm, int attribute)
		{
			this.algorithm = algorithm;
			this.attribute = attribute;
		}

		public void run()
		{
			initializeGPU(algorithm, attribute, Data.numClasses(), Data.numAttributes(), Data.numInstances());
		}
	}
}