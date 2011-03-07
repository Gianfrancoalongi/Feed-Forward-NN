/*	-----------------------------------------
	The Artificial Neural Network class
		by Gianfranco Alongi AKA zenon
	20080118	gianfranco@alongi.se
	-----------------------------------------
*/

#ifndef ANN_H
#define ANN_H

#include <vector>
using namespace std;

/**
	These are necessary typedefs for this to work properly
*/
typedef vector<double>	row;
typedef vector<row>	weight_matrix;


/**
	The ANN class is the ANN implementation as given in the course AI-2.
	It has the standard functions run, (feed forward computation) and
	backProp (a backpropagation step).

	When a sufficiently good network has been found, the network
	can be printed to a file or stdout with the netToCode() call
	which outputs the network weights in C++ code.

	@short	An ANN implementation using vector class
	@author	Gianfranco Alongi.
	
*/
class ANN 
{
	public:
		/**
			The basic empty constructor assumes 3 input neurons, 2 hidden
			and one output.
			The weights are randomized.
		*/
		ANN();

		/**
			Constructor which takes a prefabricated weight matrix for
			weights from input to hidden, and from hidden to output.
			The weightmatrix w_IH is used as follows: w_IH[h_i][i_n] 
			where h_i is hidden neuron i and i_n is input neuron n.
			The same goes for the weight matrix w_HO, which is used as
			w_HO[o_i][h_n], o_i being the i:th output neuron. And h_n being
			the n:th hidden neuron.
			The weights are not randomized.

			All integer parameters are assumed to be > 0.
		*/
		ANN(weight_matrix w_IH, weight_matrix w_HO, int Nin,	//Constructor with given matrices
			int Nhidd, int Nout);
		/**
			Constructor which takes the amout of neurons in each layer.
			The weights are randomized.
			All integer parameters are assumed to be > 0.
		*/
		ANN(int Nin,int Nhidd,int Nout);

		/**	
			A destructor for the sake of it
		*/
		~ANN();

		/**
			Returns the number of input neurons
		*/
		int 	getNin();

		/**	
			Returns the number of hidden neurons
		*/
		int 	getNhidden();

		/**
			Returns the number of output neurons for this network
		*/
		int 	getNout();

		/**
			Runs the ANN with the given input vector and returns
			the output neurons result as a vector.
			The c parameter is the sigmoid C constant used for this.
		*/
		row	run(row input,
			      double c);

		/**
			Performs a full backpropagation step for the ANN using
			the specified input vector and target output vector.
			This function updates the weights and the error of
			the network.
			The eta parameter is the learning rate.
			The c parameter is the sigmoid C constant.
		*/
		void	backProp(row input,
				 row target,
				 double eta,
				 double c);

		/**
			This method scrambles the network weights to values
			in the range [-2.9, 2.9]
		*/
		void	reset();

		/**
			Returns the error of the last run, defined as
			\frac{1}{2} \sum_{i=1}^{n(out)} err_i^2
		*/
		double 	mserror();

		/**	
			Return the weight-matrix of the weights input to hidden
			neurons.
		*/
		weight_matrix	weightsIH();

		/**
			Return the weight-matrix of the weights hidden to ouput
			neurons.
		*/
		weight_matrix	weightsHO();

		/**	
			The sigmoid function, used for internal calculation
		*/
		double	sigma(double z, double c);

		/**
			Creates a string which is the C++ code for the ANN weights.
			It also codes the amount of neurons.
		*/
		string	netToCode();
	private:
		double	c;				//The sigma C parameter
		double	error;				//Mean Square Error for last run
		int	nin;				//Number of input neurons
		int	nhidd;				//Number of hidden neurons
		int 	nout;				//Number of output neurons
		weight_matrix	wIH;				//Weights I->H
		weight_matrix	wHO;				//Weights H->O

		//This is for momentum term.
		weight_matrix	dw_wIH;				//Old weight update term (previous step)
		weight_matrix	dw_wHO;				//Old weight update term (previous step)
		double	epsylon;			//The factor
		
		void	stdSetup();			//Some standard setup things
};

#endif
