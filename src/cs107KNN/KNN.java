package cs107KNN;
import java.util.Scanner;

public class KNN {
	public static void main(String[] args) {
		// Data input for the table
		System.out.print("Hauteur du tableau de test - ");
		int height = input();
		System.out.print("Largeur du tableau de test - ");
		int width = input();
		int tests = height*width;
		System.out.println("Le nombre d'échantillons pour le test est : " + tests);
		System.out.print("Entrez le nombre d'image les plus proches à utiliser - ");
		int k = input();
		
		// Datasets used for the table
	
		System.out.println("=== Test predictions ===");
		byte[][][] imagesTrain = KNN.parseIDXimages(Helpers.readBinaryFile("datasets/5000-per-digit_images_train"));
		byte[] labelsTrain = KNN.parseIDXlabels(Helpers.readBinaryFile("datasets/5000-per-digit_labels_train"));

		byte[][][] imagesTest = KNN.parseIDXimages(Helpers.readBinaryFile("datasets/10k_images_test"));
		byte[] labelsTest = KNN.parseIDXlabels(Helpers.readBinaryFile("datasets/10k_labels_test"));

		// Data processing
		byte[] predictions = new byte[tests];
		long start = System.currentTimeMillis() ;
			for (int i = 0; i < tests; i++) {
				predictions[i] = KNN.knnClassify(imagesTest[i], imagesTrain, labelsTrain, k);
				System.out.println("Image traitée : " + (i+1) + "/" + tests);
			}
			long end = System.currentTimeMillis() ;
			double time = (end - start) / 1000d ;
			System.out.println("Time = " + time + " seconds") ;
			System.out.println("Time per test image = " + (time / tests)) ;
		Helpers.show("Test predictions", imagesTest, predictions, labelsTest, height, width);
 
		// Accuracy calculation
		System.out.println("La précision est de : " + (accuracy(predictions, labelsTest)*100) + "%");
 
	}
 
// Scanner to input an integer
public static int input() {
	int a = 0;
	Scanner keyb = new Scanner(System.in);
	do { 
		System.out.print("Entrez un entier strictement positif : ");
		a = keyb.nextInt();
		System.out.print("\n ");
	} while(a <= 0);
return a;
}
/**
* Composes four bytes into an integer using big endian convention.
*
* @param bXToBY The byte containing the bits to store between positions X and Y
* 
* @return the integer having form [ b31ToB24 | b23ToB16 | b15ToB8 | b7ToB0 ]
*/
public static int extractInt(byte b31ToB24, byte b23ToB16, byte b15ToB8, byte b7ToB0) {
	int extractedInteger = (int)((b31ToB24 & 0xFF)*Math.pow(2, 24)
								+(b23ToB16 & 0xFF)*Math.pow(2, 16)
								+(b15ToB8 & 0xFF)*Math.pow(2, 8)
								+(b7ToB0 & 0xFF));
return extractedInteger;
}

/**
* Parses an IDX file containing images
*
* @param data the binary content of the file
*
* @return A tensor of images
*/
public static byte[][][] parseIDXimages(byte[] data) {
	int magicNumber = extractInt(data[0], data[1], data[2], data[3]);
        if (magicNumber == 2051) 
            {    
                int nbrImages = extractInt(data[4], data[5], data[6], data[7]);
                int nbrRows = extractInt(data[8], data[9], data[10], data[11]);
                int nbrColumns = extractInt(data[12], data[13], data[14], data[15]);
                byte[][][] tensor = new byte[nbrImages][nbrRows][nbrColumns];
        
                    for(int i = 0; i < nbrImages; ++i) {
                        int offset = i * nbrColumns * nbrRows ;
                        for(int j = 0; j < nbrRows; ++j) {
                            for(int k = 0; k < nbrColumns ; ++k) {
                                byte pixelValue = (byte)(data[k+16+ j*nbrRows + offset]-128);
                                tensor[i][j][k] = pixelValue;
                            }
                        }
                    }
            return tensor;
            }
        else {
        return null;
        }
}

/**
* Parses an idx images containing labels
*
* @param data the binary content of the file
*
* @return the parsed labels
*/
public static byte[] parseIDXlabels(byte[] data) {
	int magicNumber = extractInt(data[0], data[1], data[2], data[3]);
        if (magicNumber == 2049)
        {
            int nbrLabels = extractInt(data[4], data[5], data[6], data[7]);
            byte[] label = new byte[nbrLabels];
                for(int i = 0; i < nbrLabels; ++i) {
                    label[i]= data[i+8];
                }
        return label;
        }
        else {
        return null;
        }
}

/**
* @brief Computes the squared L2 distance of two images
* 
* @param a, b two images of same dimensions
* 
* @return the squared euclidean distance between the two images
*/
public static float squaredEuclideanDistance(byte[][] a, byte[][] b) {
	float eSquare = 0;
	for(int i = 0; i < a.length; ++i) {
		for(int j = 0; j<a[i].length; ++j) {
			eSquare += (a[i][j]-b[i][j])*(a[i][j]-b[i][j]);
		}
	}
return eSquare;
}

/**
* @brief Computes the inverted similarity between 2 images.
* 
* @param a, b two images of same dimensions
* 
* @return the inverted similarity between the two images
*/
public static float invertedSimilarity(byte[][] a, byte[][] b) {
	float si = 0;
	float innerProduct = 0;
	float standardDeviation = 0;
	float sdA = 0;
	float sdB = 0;
	float meanA = meanImage(a);
	float meanB = meanImage(b);
	for(int i = 0; i < a.length; ++i) {
		for(int j = 0; j<a[0].length; ++j) {
			innerProduct += (a[i][j]-meanA)*(b[i][j]-meanB);
			sdA += (a[i][j]-meanA)*(a[i][j]-meanA);
			sdB += (b[i][j]-meanB)*(b[i][j]-meanB);
		}
	}
	standardDeviation = (float)Math.sqrt(sdA*sdB);
	si = 1 - innerProduct/standardDeviation;
return si;
}
 
// Average values ​​of the image's pixels
public static float meanImage(byte[][] a) {
	float mean = 0;
	for(int i = 0; i < a.length; ++i) {
		for(int j = 0; j<a[0].length; ++j) {
			mean += a[i][j];
		}
	}
	mean /= (a.length*a[0].length);
return mean;
}
 
 

/**
* @brief Quicksorts and returns the new indices of each value.
* 
* @param values the values whose indices have to be sorted in non decreasing
*               order
* 
* @return the array of sorted indices
* 
*         Example: values = quicksortIndices([3, 7, 0, 9]) gives [2, 0, 1, 3]
*/
public static int[] quicksortIndices(float[] values) {
	int[] indices = new int[values.length];
	for(int i = 0; i < values.length; ++i) {
		indices[i] = i;
	}
	quicksortIndices(values, indices, 0, values.length-1);
return indices;
}

/**
* @brief Sorts the provided values between two indices while applying the same
*        transformations to the array of indices
* 
* @param values  the values to sort
* @param indices the indices to sort according to the corresponding values
* @param         low, high are the **inclusive** bounds of the portion of array
*                to sort
*/
public static void quicksortIndices(float[] values, int[] indices, int low, int high) {
	int l = low;
	int h = high;
	float pivot = values[low];
	while(l <= h ) {
		if(values[l] < pivot) {
			++l;
		} 
		else if(values[h] > pivot) {
			--h;
		}
		else {
			swap(l, h, values, indices);
			++l;
			--h;
		}
	}
	if(low < h) {
		quicksortIndices(values, indices, low, h);
	}
	if(high > l) {
		quicksortIndices(values, indices, l, high);
	}
}

/**
* @brief Swaps the elements of the given arrays at the provided positions
* 
* @param         i, j the indices of the elements to swap
* @param values  the array floats whose values are to be swapped
* @param indices the array of ints whose values are to be swapped
*/
public static void swap(int i, int j, float[] values, int[] indices) {
	//Swap values
	float temp1 = values[i];
	values[i] = values[j];
	values[j] = temp1;
 
	//Swap indices
	int temp2 = indices[i];
	indices[i] = indices[j];
	indices[j] = temp2;
	}

/**
* @brief Returns the index of the largest element in the array
* 
* @param array an array of integers
* 
* @return the index of the largest integer
*/
public static int indexOfMax(int[] array) {
	float[] values = new float[array.length];
	for( int i = 0; i < array.length; ++i) {
		values[i] = array[i];
	}
	int[] indice = quicksortIndices(values);
	int maxIndex = indice[values.length-1];
return maxIndex;
}

/**
* The k first elements of the provided array vote for a label
*
* @param sortedIndices the indices sorted by non-decreasing distance
* @param labels        the labels corresponding to the indices
* @param k             the number of labels asked to vote
*
* @return the winner of the election
*/
public static byte electLabel(int[] sortedIndices, byte[] labels, int k) {
	int[] frequencies = new int[10];
	int max = 0;
	for(int i = 0; i < k; ++i) {
		++frequencies[labels[sortedIndices[i]]];
	}
	max = indexOfMax(frequencies);
return (byte)max;
}

/**
* Classifies the symbol drawn on the provided image
*
* @param image       the image to classify
* @param trainImages the tensor of training images
* @param trainLabels the list of labels corresponding to the training images
* @param k           the number of voters in the election process
*
* @return the label of the image
* You can change invertedSimilarity to squaredEuclideanDistance if you want
*/
public static byte knnClassify(byte[][] image, byte[][][] trainImages, byte[] trainLabels, int k) {
	float[] distances = new float[trainImages.length];
	for(int i = 0; i<trainImages.length; ++i) {
		distances[i] = invertedSimilarity(image, trainImages[i]);
	}
return electLabel(quicksortIndices(distances), trainLabels, k);
}

/**
* Computes accuracy between two arrays of predictions
* 
* @param predictedLabels the array of labels predicted by the algorithm
* @param trueLabels      the array of true labels
* 
* @return the accuracy of the predictions. Its value is in [0, 1]
*/
public static double accuracy(byte[] predictedLabels, byte[] trueLabels) {
	double a = 0;
	for(int i = 0; i < predictedLabels.length; ++i) {
		if(predictedLabels[i] == trueLabels[i]) {
			++a;
		}
	}
return (a/predictedLabels.length);
}
}