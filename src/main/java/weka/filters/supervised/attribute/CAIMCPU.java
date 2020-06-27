/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * CAIM.java
 * Copyright (C) 2012 Dat T Nguyen
 */ 
package weka.filters.supervised.attribute;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Vector;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.SparseInstance;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;

/**
<!-- globalinfo-start -->
 * CAIM Discretization Algorithm.<br/>
 * <br/>
 * For more information see:<br/>
 * <br/>
 * L. A. Kurgan, K. J. Cios: CAIM Discretization Algorithm. IEEE Transactions On Knowledge And Data Engineering, Vol. 16, No. 2, 145-153, 2004.
 * <p/>
<!-- globalinfo-end -->
 * 
<!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * @article{"",
 *    author = {L. A. Kurgan and K. J. Cios},
 *    journal = {IEEE Transactions On Knowledge And Data Engineering},
 *    volume = {16},
 *    number = {2},
 *    pages = {145-153},
 *    title = {{CAIM Discretization Algorithm}},
 *    year = {2004}
 * }
 * </pre>
 * <p/>
<!-- technical-bibtex-end -->
 * 
<!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -R &lt;list of columns&gt;
 *  List contains columns to be discretized (default first-last)</pre>
 * 
 * <pre> -O &lt;Output format&gt;
 *  Choose format of output (default false)</pre>
 * 
 * <pre> -C &lt;Name of class's column&gt;
 *  Name of column contains class (default class)</pre>
 * 
<!-- options-end -->
 *
 * @author Dat Nguyen (nguyendt22@vcu.edu)
 */ 

public class CAIMCPU extends Filter implements SupervisedFilter, OptionHandler
{
	private static final long serialVersionUID = 1L;

	//Stores which columns to discretize 
	protected Range m_DiscretizeCols = new Range();
	//Output in which format 0,1,2... or [xx,yy)  
	protected boolean m_OutputInNumeric=false; 
	//Store name of class's column
	protected String m_ClassName="class";
	//Discretized Scheme's result
	protected ArrayList<ArrayList<Double>> SchemeList;
	//List of class's value
	protected ArrayList<String> ClassValueList;
	//Position of class's column
	protected int Index=-1 ;

	//---------------------------------------------------------------------------
	protected double CalculateCAIM(Vector<Double> D,double pointvalue,Object[] value, int[][]Appearance)
	{
		ArrayList<Double> tempD= new ArrayList<Double>(D);
		tempD.add(pointvalue);
		Object[] tempSort= tempD.toArray();
		Arrays.sort(tempSort);
		int TotalCol=tempSort.length-1;
		int TotalRow=ClassValueList.size()+1;
		int [][] QuantaMatrix= new int[TotalRow][TotalCol];
		for (int i=0;i<TotalCol;i++)
		{
			double min=Double.parseDouble(tempSort[i].toString());
			double max=Double.parseDouble(tempSort[i+1].toString());
			int left=0, right=0;
			for (;Double.parseDouble(value[left].toString())<min;left++);
			for (;right < value.length && Double.parseDouble(value[right].toString())<=max;right++);
			int SumCol=0;
			for (int c=0;c<TotalRow-1;c++)
			{
				int sum=0;
				for (int col=left;col<right;col++) sum+=Appearance[c][col];
				QuantaMatrix[c][i]=sum;
				SumCol+=sum;
			}
			QuantaMatrix[TotalRow-1][i]=SumCol;	  
		}

		double CAIMvalue=0;
		for (int col=0;col<TotalCol;col++)
		{
			long max=0;
			for (int c=0;c<TotalRow-1;c++) max=QuantaMatrix[c][col]>max?QuantaMatrix[c][col]:max;
			double res = max / (double)QuantaMatrix[TotalRow-1][col];
			CAIMvalue+= max * res;
		}
		CAIMvalue=CAIMvalue/(tempSort.length-1);
		return CAIMvalue;
	}
	//---------------------------------------------------------------------------
	protected ArrayList<Double> CAIM (Instances Table, int column,int Class)
	{
		ArrayList<Double> Scheme=new ArrayList<Double>();
		Vector<Double> temp= new Vector<Double>();
		for (int i=0;i<Table.numInstances();i++)
			if (temp.indexOf(Table.instance(i).value(column))<0)
				temp.add(Table.instance(i).value(column));
		Object[] value= temp.toArray();
		Arrays.sort(value);
		int[][] Appearance = new int[ClassValueList.size()][value.length];
		for (int i=0;i<Table.numInstances();i++)
		{
			int row=ClassValueList.indexOf(Table.instance(i).stringValue(Class));
			int col=Arrays.binarySearch(value, Table.instance(i).value(column));
			Appearance[row][col]++;
		}
		Vector<Double> D = new Vector<Double>();
		D.add((Double) value[0]);
		D.add((Double) value[value.length-1]);
		Vector<Double> B= new Vector<Double>();
		for (int i=0;i<=value.length-2;i++)
			B.add((Double.parseDouble(value[i].toString())+Double.parseDouble(value[i+1].toString()))/2);
		double GlobalCAIM=0;
		int step=1;
		int LastStep=0;
		LastStep=ClassValueList.size();
		boolean notdone=true;
		if (B.size()==0) notdone=false;
		while(notdone)
		{
			double MaxCAIM=0;
			int midpoint=-1;
			for (int pos=0;pos<B.size();pos++)
			{
				double CurrentCAIM=CalculateCAIM(D,Double.parseDouble(B.get(pos).toString()),value,Appearance);

				if (CurrentCAIM>MaxCAIM)
				{
					MaxCAIM=CurrentCAIM;
					midpoint=pos;
				}
			}

			if (midpoint==-1) break;

			if (!CheckConditionStop(MaxCAIM, GlobalCAIM,step,LastStep))
			{
				GlobalCAIM=MaxCAIM;
				D.add(B.get(midpoint));
				B.remove(midpoint);
				step++;
			}
			else
				break;
			if (B.size()==0) break;
		}
		Object[] tempSort= D.toArray();
		Arrays.sort(tempSort);
		for (int j=0;j<ClassValueList.size();j++)
		{
			for (int i=1;i<tempSort.length;i++)
			{
				double min=	(Double)tempSort[i-1];
				double max=	(Double)tempSort[i];
				int left=0, right=value.length-1;
				while ((Double)value[left]<min)left++;
				while ((Double)value[right]>max)right--;
			}
		}
		for (int i=0;i<tempSort.length;i++)Scheme.add(Double.parseDouble(tempSort[i].toString()));
		return Scheme;
	}
	//---------------------------------------------------------------------------  
	protected boolean CheckConditionStop(double MaxCAIM, double GlobalCAIM,int step, int LastStep)
	{
		if (MaxCAIM>GlobalCAIM||step<LastStep) return false; else return true;
	}
	//-----------------------------------------------------------------------------
	protected void setOutputFormat() {
		Instances Data = getInputFormat();
		ArrayList<Attribute> attributes = new ArrayList<Attribute>(Data.numAttributes());
		for(int current = 0; current < Data.numAttributes(); current++)
		{
			if ((!Data.attribute(current).isNumeric())||(!m_DiscretizeCols.isInRange(current))||(current==Index) )
			{ 
				attributes.add((Attribute)Data.attribute(current).copy());
				continue;
			}
			ArrayList<String> attribValues = new ArrayList<String>(1);
			ArrayList<Double> l = SchemeList.get(current);
			for (int i=0;i<l.size()-1;i++)
				if(m_OutputInNumeric)
					attribValues.add(Integer.toString(i));
				else
				{
					String s="["+l.get(i).toString()+"-"+l.get(i+1).toString()+")";
					if (i==(l.size()-2))
						s=s.replace(")", "]");
					attribValues.add(s);
				}
			attributes.add(new Attribute(Data.attribute(current).name(),attribValues));

		}//end for current
		Instances outputFormat =  new Instances(Data.relationName(), attributes, 0);
		setOutputFormat(outputFormat);
	}
	//---------------------------------------------------------------------------
	protected void convertInstance(Instance instance) {
		double [] vals = new double [outputFormatPeek().numAttributes()];
		Instances Data = getInputFormat();
		for(int current = 0; current < Data.numAttributes(); current++)
		{
			if ((!Data.attribute(current).isNumeric())||(!m_DiscretizeCols.isInRange(current))||(current==Index) )
				vals[current]=instance.value(current);
			else
			{
				ArrayList<Double> l = SchemeList.get(current);
				int k=0;
				while (instance.value(current)> Double.parseDouble(l.get(k).toString()))
				{
					k++;
					if (k==l.size())break;
				}
				k--;
				if (k<0)k=0;
				if(k == l.size()-1)	k--;
				vals[current]=k;
			}
		}//end for current
		Instance inst = null;
		if (instance instanceof SparseInstance) {
			inst = new SparseInstance(instance.weight(), vals);
		} else {
			inst = new DenseInstance(instance.weight(), vals);
		}
		inst.setDataset(getOutputFormat());
		copyValues(inst, false, instance.dataset(), getOutputFormat());
		inst.setDataset(getOutputFormat());
		push(inst);
	}

	protected Instance convertInstanceTest(Instance instance) {
		double [] vals = new double [outputFormatPeek().numAttributes()];
		Instances Data = getInputFormat();
		for(int current = 0; current < Data.numAttributes(); current++)
		{
			if ((!Data.attribute(current).isNumeric())||(!m_DiscretizeCols.isInRange(current))||(current==Index) )
				vals[current]=instance.value(current);
			else
			{
				ArrayList<Double> l = SchemeList.get(current);
				int k=0;
				while (instance.value(current)> Double.parseDouble(l.get(k).toString()))
				{
					k++;
					if (k==l.size())break;
				}
				k--;
				if (k<0)k=0;
				if(k == l.size()-1)	k--;
				vals[current]=k;
			}
		}//end for current
		Instance inst = null;
		if (instance instanceof SparseInstance) {
			inst = new SparseInstance(instance.weight(), vals);
		} else {
			inst = new DenseInstance(instance.weight(), vals);
		}
		return inst;
	}
	//---------------------------------------------------------------------------
	public boolean batchFinished()  throws Exception
	{
		Instances Data=  getInputFormat();
		SchemeList= new ArrayList<>(Data.numAttributes());
		Attribute ClassAttribute=Data.attribute(Data.numAttributes()-1);
		Object o= ClassAttribute;
		if (o ==null)
		{ throw new Exception("Wrong name in class's attribute");}
		Index=ClassAttribute.index();
		Enumeration<Object> enu=ClassAttribute.enumerateValues();
		ClassValueList= new ArrayList<String>();
		while(enu.hasMoreElements())ClassValueList.add((String)enu.nextElement());

		for (int current=0;current<Data.numAttributes();current++)
			SchemeList.add(new ArrayList<Double>());

		ExecutorService threadExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

		for (int current=0;current<Data.numAttributes();current++)
		{
			if (current==Index) { continue;}
			if (!Data.attribute(current).isNumeric()){ continue;}
			if (!m_DiscretizeCols.isInRange(current)) { continue;}

			threadExecutor.execute(new evaluationThread(Data, current));
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

		if (getInputFormat() == null) {throw new IllegalStateException("No input instance format defined");}
		setOutputFormat();
		for(int i = 0; i < getInputFormat().numInstances(); i++)
			convertInstance(getInputFormat().instance(i));
		return true;
	}
	//---------------------------------------------------------------------------
	public String globalInfo() {return "An instance filter that discretizes a range of numeric";}
	//---------------------------------------------------------------------------
	public String outputInNumericTipText(){ return "true:output in 1,2,3, false:output in [a,b),[c,d), format";}
	//---------------------------------------------------------------------------
	public boolean getOutputInNumeric(){return m_OutputInNumeric;}
	//---------------------------------------------------------------------------
	public void setOutputInNumeric(boolean val){m_OutputInNumeric=val;}
	//---------------------------------------------------------------------------
	public String classNameTipText(){ return "Input column's name contain class";}
	//---------------------------------------------------------------------------
	public String getClassName(){return m_ClassName;}
	//---------------------------------------------------------------------------
	public void setClassName(String val){m_ClassName=val;}
	//---------------------------------------------------------------------------
	public String attributeIndicesTipText() {
		return "Specify range of attributes to act on."
				+ " This is a comma separated list of attribute indices, with"
				+ " \"first\" and \"last\" valid values. Specify an inclusive"
				+ " range with \"-\". E.g: \"first-3,5,6-10,last\".";
	}
	//---------------------------------------------------------------------------
	public String getAttributeIndices() {return m_DiscretizeCols.getRanges();}
	//---------------------------------------------------------------------------
	public void setAttributeIndices(String rangeList){m_DiscretizeCols.setRanges(rangeList); }
	//---------------------------------------------------------------------------
	public void setAttributeIndicesArray(int [] attributes) {setAttributeIndices(Range.indicesToRangeList(attributes));}
	//---------------------------------------------------------------------------
	public CAIMCPU() { setAttributeIndices("first-last");}
	//---------------------------------------------------------------------------
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>();
		newVector.addElement(new Option(
				"\tSpecifies list of columns to Discretize. First"
						+ " and last are valid es.\n"
						+ "\t(default none)",
						"R", 1, "-R <col1,col2-col4,...>"));
		newVector.addElement(new Option("\tOutput in numeric format.","O", 1, "-O"));
		newVector.addElement(new Option("\tClass column's name.","C", 0, "-C"));
		return newVector.elements();
	}
	//---------------------------------------------------------------------------
	public void setOptions(String[] options) throws Exception {
		setOutputInNumeric(Utils.getFlag('O', options));
		String convertList = Utils.getOption('R', options);
		if (convertList.length() != 0)
			setAttributeIndices(convertList);
		else 
			setAttributeIndices("first-last");
		String ClassName=Utils.getOption('C', options);
		m_ClassName=ClassName;
		if (getInputFormat() != null) setInputFormat(getInputFormat());
	}
	//---------------------------------------------------------------------------
	public String [] getOptions() {

		String [] options = new String [20];
		int current = 0;
		options[current++] = "-O"; options[current++] = "" + m_OutputInNumeric;
		options[current++] = "-C"; options[current++] = "" + m_ClassName;
		if (!getAttributeIndices().equals(""))
			options[current++] = "-R"; options[current++] = getAttributeIndices();
			while (current < options.length)  options[current++] = "";
			return options;
	}
	//---------------------------------------------------------------------------
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();
		result.enableAllAttributes();
		result.enable(Capability.UNARY_CLASS);
		result.enable(Capability.NOMINAL_CLASS);
		return result;
	}
	//---------------------------------------------------------------------------
	public boolean setInputFormat(Instances instanceInfo) throws Exception {
		super.setInputFormat(instanceInfo);
		m_DiscretizeCols.setUpper(instanceInfo.numAttributes() - 1);
		return false;
	}
	//---------------------------------------------------------------------------
	public boolean input(Instance instance) {
		if (getInputFormat() == null) {throw new IllegalStateException("No input instance format defined");  }
		bufferInput(instance);
		return false;
	}
	//---------------------------------------------------------------------------

	/////////////////////////////////////////////////////////////////
	// -------------------------------------------- Evaluation Thread
	/////////////////////////////////////////////////////////////////

	private class evaluationThread extends Thread
	{
		private int attribute;
		private Instances Data;

		public evaluationThread(Instances Data, int attribute)
		{
			this.Data = Data;
			this.attribute = attribute;
		}

		public void run()
		{
			SchemeList.set(attribute, new ArrayList<Double>(CAIM(Data,attribute,Index)));
		}
	}
}
