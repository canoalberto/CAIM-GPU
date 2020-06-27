#include "parameters.h"

__global__ void computeCAIMValues(float* caimValues, int* appearance, int* intervals, int numberIntervals, int numberClasses, int numbermidPoints);

JNIEXPORT void JNICALL
Java_weka_filters_supervised_attribute_CAIMGPU_initializeGPU(JNIEnv *env, jobject obj, jobject algorithm, jint attribute, jint numberClasses, jint numberAttributes, jint numberInstances)
{
	int deviceCount, numberValues, numberIntervals, numbermidPoints, iteration = 1;
	int *h_appearance, *d_appearance, *h_classValues, *d_classValues;
	float *h_attributeValues, *d_attributeValues, *h_midpoints;
	int *h_intervals, *d_intervals, *tempInterval, *aux;
	float *h_caimValues, *d_caimValues;
	float globalCAIM = 0.0f;
	
	// Set the GPU device number and properties
	cudaSetDeviceFlags(cudaDeviceScheduleSpin);
	cudaGetDeviceCount(&deviceCount);
	
	cudaSetDevice(attribute % deviceCount);

	jclass cls = env->GetObjectClass(algorithm);
	jmethodID getAttributeValues = env->GetMethodID(cls, "getAttributeValues", "(I)[F");
	jmethodID getClassValues = env->GetMethodID(cls, "getClassValues", "()[I");
	jmethodID addInterval = env->GetMethodID(cls, "addInterval", "(IF)V");
	
	jfloatArray jattributeValues = (jfloatArray) env->CallObjectMethod(algorithm, getAttributeValues, attribute);
	jintArray jclassValues = (jintArray) env->CallObjectMethod(algorithm, getClassValues);
	
	h_attributeValues = (float*) env->GetFloatArrayElements(jattributeValues, 0);
	h_classValues = (int*) env->GetIntArrayElements(jclassValues, 0);
	
	h_appearance = (int*) calloc(numberClasses * numberInstances, sizeof(int)); 
	cudaMalloc((void**) &d_attributeValues, numberInstances * sizeof(float));
	cudaMalloc((void**) &d_classValues, numberInstances * sizeof(int));
	
	cudaMemcpy(d_attributeValues, h_attributeValues, numberInstances * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy(d_classValues, h_classValues, numberInstances * sizeof(int), cudaMemcpyHostToDevice );
	
	thrust::device_ptr<float> d_attributeValues_ptr = thrust::device_pointer_cast(d_attributeValues);
	thrust::device_ptr<int>   d_classValues_ptr = thrust::device_pointer_cast(d_classValues);
	
	thrust::sort_by_key(d_attributeValues_ptr, d_attributeValues_ptr + numberInstances, d_classValues_ptr);
	
	cudaMemcpy(h_attributeValues, d_attributeValues, numberInstances * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy(h_classValues, d_classValues, numberInstances * sizeof(int), cudaMemcpyDeviceToHost );
	
	float currentValue = h_attributeValues[0];
	int offset = 0;
	numberValues = 1;
	
	for(int i = 0; i < numberInstances; i++)
	{
		if(currentValue != h_attributeValues[i])
		{
			offset++;
			
			currentValue = h_attributeValues[i];
			
			h_attributeValues[offset] = h_attributeValues[i];
			
			numberValues++;
		}
		
		h_appearance[offset*numberClasses + h_classValues[i]]++;
	}
	
	numberIntervals = 2;
	numbermidPoints = numberValues - 1;
	int numbermidPointsRemaining = numbermidPoints;
	
	h_intervals = (int*) calloc(numberClasses + 2, sizeof(int));
	tempInterval = (int*) calloc(numberClasses + 2, sizeof(int));
	h_midpoints = (float*) calloc(numberValues - 1, sizeof(float));
	h_caimValues = (float*) malloc(numbermidPoints * sizeof(float));
	
	cudaMalloc((void**) &d_intervals, (numberClasses + 2) * sizeof(int));
	cudaMalloc((void**) &d_appearance,  numberClasses * numberValues * sizeof(int));
	cudaMalloc((void**) &d_caimValues, numbermidPoints * sizeof(float));
	
	cudaMemcpy(d_appearance, h_appearance, numberClasses * numberValues * sizeof(int), cudaMemcpyHostToDevice );
		
	h_intervals[0] = 0;
	h_intervals[1] = numberValues-1;
	
	for (int i = 0; i < numbermidPoints; i++)
		h_midpoints[i] = (h_attributeValues[i] + h_attributeValues[i+1]) / 2.0f;
	
	dim3 threadsCAIMValues(THREADS_EVAL_BLOCK, 1);
	dim3 gridCAIMValues((int) ceil(numbermidPoints / (THREADS_EVAL_BLOCK * 1.0f)), 1);
	
	while(1)
	{
		cudaMemcpy(d_intervals, h_intervals, numberIntervals * sizeof(int), cudaMemcpyHostToDevice );
		
		computeCAIMValues <<< gridCAIMValues, threadsCAIMValues >>> (d_caimValues, d_appearance, d_intervals, numberIntervals-1, numberClasses, numbermidPoints);
		
		cudaMemcpy(h_caimValues, d_caimValues, numbermidPoints * sizeof(float), cudaMemcpyDeviceToHost );
		
		int bestmidPoint = -1;
		float bestCAIM = -1;
		
		for(int i = 0; i < numbermidPoints; i++)
		{
			if(h_caimValues[i] > bestCAIM)
			{
				bestCAIM = h_caimValues[i];
				bestmidPoint = i;
			}
		}
		
		if (bestmidPoint == -1)		break;
		
		if(bestCAIM > globalCAIM || iteration < numberClasses)
		{
			globalCAIM = bestCAIM;
			
			for(int i = 0; i < numberIntervals; i++)
			{
				tempInterval[i] = h_intervals[i];
				
				if((bestmidPoint+1) <= h_intervals[i])
				{
					tempInterval[i] = (bestmidPoint+1);
					
					for(int j = i; j < numberIntervals; j++)
					{
						tempInterval[j+1] = h_intervals[j];
					}
					
					break;
				}
			}
			
		    aux = h_intervals;
		    h_intervals = tempInterval;
		    tempInterval = aux;
			
			iteration++;
			numberIntervals++;
			numbermidPointsRemaining--;
		}
		else
			break;
			
		if (numbermidPointsRemaining == 0) break;
	}
	
	env->CallVoidMethod(algorithm, addInterval, attribute, h_attributeValues[0]);
	
	for(int i = 1; i < numberIntervals - 1; i++)
		env->CallVoidMethod(algorithm, addInterval, attribute, h_midpoints[h_intervals[i]-1]);
		
	env->CallVoidMethod(algorithm, addInterval, attribute, h_attributeValues[numberValues-1]);
		
	env->ReleaseFloatArrayElements(jattributeValues, h_attributeValues, 0);
	env->ReleaseIntArrayElements(jclassValues, h_classValues, 0);
	
	cudaFree(d_attributeValues);
	cudaFree(d_classValues);
	cudaFree(d_intervals);
	cudaFree(d_appearance);
	cudaFree(d_caimValues);
	
	free(h_appearance);
	free(h_caimValues);
	free(h_intervals);
	free(h_midpoints);
	free(tempInterval);
}

__device__ float calculate(int left, int right, int* appearance, int numberClasses)
{
	int columnSum [MAX_CLASSES] = {0};
		
	for(int i = left; i < right; i++)
	{
		for(int j = 0; j < numberClasses; j++)
		{
			columnSum[j] += appearance[i*numberClasses + j];
		}
	}
	
	int columnMax = 0, suma = 0;
	
	for(int j = 0; j < numberClasses; j++)
	{
		suma += columnSum[j];
		
		if(columnSum[j] > columnMax)
			columnMax = columnSum[j];
	} 
			
	float res = columnMax / (float) suma;
	res = res * columnMax;
	
	return res;
}

__global__ void computeCAIMValues(float* caimValues, int* appearance, int* intervals, int numberIntervals, int numberClasses, int numbermidPoints)
{
	int midPoint = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(midPoint < numbermidPoints)
	{
		float CAIMValue = 0.0f;
		
		for(int k = 0; k < numberIntervals; k++)
		{
			int left = intervals[k];
			int right = intervals[k+1];
			
			if((midPoint+1) == left)
			{
				caimValues[midPoint] = 0.0f;
				return;
			}
			
			if(k == numberIntervals-1)	right++;
			
			if(left <= midPoint && midPoint < right)
			{
				CAIMValue += calculate(left, midPoint+1, appearance, numberClasses);
				CAIMValue += calculate(midPoint+1, right, appearance, numberClasses);
			}
			else
			{
				CAIMValue += calculate(left, right, appearance, numberClasses);
			}
		}
		
		caimValues[midPoint] = CAIMValue / (float) (numberIntervals+1);
	}
}