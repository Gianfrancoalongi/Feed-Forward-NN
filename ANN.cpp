
/*	++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	ANN implementation with momentum term for faster convergence.
	Gianfranco Alongi AKA zenon
		gianfranco@alongi.se			20080120
	++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
*/

#include "ANN.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <sstream>
using namespace std;

/*	A destructor, just to see that things are ending nice	*/
ANN :: ~ANN()
{
	cout << "Destructing the ANN" << endl;
}

/*	An empty constructor, just to make sure there is one	*/
ANN :: ANN()
{
	//Make the matrices big enough
	nin	= 3;
	nhidd	= 2;
	nout	= 1;

	stdSetup();
	reset();
}



/* Constructor with given weight matrices, does not randomize weights	*/
ANN :: ANN(weight_matrix w_IH, weight_matrix w_HO, int Nin, int Nhidd, int Nout)
{
	nin	= Nin;
	nhidd	= Nhidd;
	nout	= Nout;

	stdSetup();	

	wIH	= w_IH;
	wHO	= w_HO;
	
}

/*	A constructor which takes some dimensions, randomizes the weights	*/
ANN :: ANN(int Nin,int Nhidd,int Nout)
{
	//Make the matrices big enough
	nin	= Nin;
	nhidd	= Nhidd;
	nout	= Nout;

	stdSetup();	//Weights, error, delta_w, error sign,
	reset();	//Now randomize the weights, 

}


/*	Std setup utility	*/
void ANN :: stdSetup()
{
	//Create the weight matrices
	wIH.resize(nhidd);
	for(int h_n=0; h_n < nhidd; h_n++){
		wIH[h_n].resize(nin+1);	//Add one for bias weight
	}
	wHO.resize(nout);
	for(int o_n=0; o_n < nout; o_n++){
		wHO[o_n].resize(nhidd+1);	//Add one for bias weight
	}

	//Set error to some huge value
	error = pow(2.0,8.0);

	//Last weight updates are set to 0.
	epsylon	= 0.5;
	dw_wIH.resize(nhidd);
	for(int k = 0; k < nhidd; k++){	dw_wIH[k].resize(nin+1); }
	dw_wHO.resize(nout);
	for(int k = 0; k < nout; k++){	dw_wHO[k].resize(nhidd+1);}

}

/*	Some simple get-functions	*/
int ANN::getNin(){		return nin;	}
int ANN::getNhidden(){		return nhidd;	}
int ANN::getNout(){		return nout;	}
double ANN::mserror(){		return error;	}
weight_matrix ANN::weightsIH(){	return wIH;	}
weight_matrix ANN::weightsHO(){	return wHO;	}


/*	Runs the neural net with some input, and returns an output	*/
row ANN::run(row input,double c)
{
	row output(nout);

	if(input.size() != nin){ 
		cout << "Error: Input data differs from input neuron amount" << endl;
		cout << "Error code -1" << endl;
		return output;
	}

	//Calculate hidden neuron outputs
	row hidd_out(nhidd);
	for(int h_n = 0; h_n < nhidd; h_n++){
		double h_n_linsum= wIH[h_n][0];	//Bias weight
		for(int i_n = 0; i_n < nin; i_n++){
			h_n_linsum += input[i_n]*wIH[h_n][i_n+1];
		}
		hidd_out[h_n] = sigma(h_n_linsum,c); // Sigma function
	}

	//Calculate output from output neurons
	for(int o_n = 0; o_n < nout; o_n++){
		double o_n_linsum= wHO[o_n][0];	//Bias weight
		for(int h_n = 0; h_n < nhidd; h_n++){
			o_n_linsum += hidd_out[h_n]*wHO[o_n][h_n+1];
		}
		output[o_n] =  sigma(o_n_linsum,c); // Sigma function
	}

	return output;
}

/*	A full backpropagation step with momentum term	*/
void ANN::backProp(row input,row target,double eta, double c)
{
	//Check that dimensions will be correct
	if(input.size() != nin){ 
		cout << "Error: Input data differs from input neuron amount" << endl;
		cout << "Error code -2" << endl;
		cout << "Input data size: " << input.size() << " Input neurons: " << nin << endl;
		return;
	}
	
	//Make space for new weight matrices--------------------------------------
	weight_matrix wHO_new(nout);
	for(int k = 0; k < nout; k++){
		wHO_new[k].resize(nhidd+1);	//One extra weight due to bias
	}
	weight_matrix wIH_new(nhidd);
	for(int k = 0; k < nhidd; k++){
		wIH_new[k].resize(nin+1);	//Same here, one extra due to bias
	}
	
	//Calculate the outputs from each layer-----------------------------------
	row hidd_out(nhidd), out(nhidd);

	//Calculate hidden neuron outputs
	for(int h_n = 0; h_n < nhidd; h_n++){
		double h_n_linsum= wIH[h_n][0];	//Bias weight
		for(int i_n = 0; i_n < nin; i_n++){
			h_n_linsum += input[i_n]*wIH[h_n][i_n+1];
		}
		hidd_out[h_n] = sigma(h_n_linsum,c); // Sigma function
	}

	//Calculate output from output neurons
	for(int o_n = 0; o_n < nout; o_n++){
		double o_n_linsum= wHO[o_n][0];	//Bias weight
		for(int h_n = 0; h_n < nhidd; h_n++){
			o_n_linsum += hidd_out[h_n]*wHO[o_n][h_n+1];
		}
		out[o_n] =  sigma(o_n_linsum,c); // Sigma function
	}
	
	//Now calculate the error for each output neuron------------------------
	row out_error(nout);
	for(int o_n = 0; o_n < nout; o_n++){
		out_error[o_n] = target[o_n] - out[o_n];
	}

	//Calculate the weight update for each weight w_ij H->O, also update the
	//weight into the new weight matrix-------------------------------------
	//Store delta values for the kappa calculations in the hidden layer
	row delta(nout);
	double sigmaPrime, delta_i, dw_ij;
	for(int o_i = 0; o_i < nout; o_i++){

		//Calculate delta_i for this particular output neuron-----------
		sigmaPrime	= c*out[o_i]*(1 - out[o_i]);
		delta_i		= out_error[o_i]*sigmaPrime;
		delta[o_i]	= delta_i;

		//Bias weight update first of all, and store the delta.
		dw_ij		= eta*delta_i + epsylon*dw_wHO[o_i][0];
		dw_wHO[o_i][0]	= dw_ij;
		wHO_new[o_i][0]	= wHO[o_i][0] + dw_ij;

		//Now calculate delta weight and update weights H->O------------
		for(int h_j=0; h_j < nhidd; h_j++){
			dw_ij   = eta*delta_i*hidd_out[h_j] + epsylon*dw_wHO[o_i][h_j+1];
			dw_wHO[o_i][h_j+1]  = dw_ij;
			wHO_new[o_i][h_j+1] = wHO[o_i][h_j+1] + dw_ij;
		}

	}

	//Calculate the weight update for each weight w_ij I->H, also update the
	//weight into the new weight matrix-------------------------------------
	double summed_weight_i, kappa_i;
	for(int h_i = 0; h_i < nhidd; h_i++){

		//Calculate the kappa value for each hidden neuron
		summed_weight_i = 0;
		for(int o_l = 0; o_l < nout; o_l++){
			summed_weight_i += delta[o_l]*wHO[o_l][h_i+1];
		}
		sigmaPrime	= c*hidd_out[h_i]*(1 - hidd_out[h_i]);
		kappa_i		= eta*sigmaPrime*summed_weight_i;

		//Now update all I->H weights, begin with the bias weight
		dw_ij		= eta*kappa_i*1 + epsylon*dw_wIH[h_i][0];
		dw_wIH[h_i][0]	= dw_ij;
		wIH_new[h_i][0] = wIH[h_i][0] + dw_ij;

		for(int i_n = 0; i_n < nin; i_n++){
			dw_ij	= eta*kappa_i*input[i_n] + epsylon*dw_wIH[h_i][i_n+1];
			dw_wIH[h_i][i_n+1]	= dw_ij;
			wIH_new[h_i][i_n+1]	= wIH[h_i][i_n+1]+dw_ij;
		}		
	}

	//Make sure to copy the weight matrices..
	wHO = wHO_new;
	wIH = wIH_new;

	//Calculate error
	error = 0;
	for(int e = 0; e < nout; e++){
		error += pow(out_error[e],2);
	}
	error /= 2;
}

/*	Randomize the weights	*/
void ANN::reset()
{
	srand(time(0));	

	//Randomize the values in the interval given
	for(int o_n = 0; o_n < nout; o_n++){
		for(int h_n = 0; h_n < nhidd+1; h_n++){
			double a	= (double (rand() % 10)) / 100;
			double b	= ( double (rand() % 5) - 2 ) / 10;
			wHO[o_n][h_n]	= a + b;
		}
	}

	for(int h_n = 0; h_n < nhidd; h_n++){
		for(int i_n = 0; i_n < nin+1; i_n++){
			double a	= (double (rand() % 10)) / 100;
			double b	= ( double (rand() % 5) - 2 ) / 10;
			wIH[h_n][i_n]	= a + b;
		}
	}

}

/*	The sigma function	*/
double ANN::sigma(double z, double c)
{
	return 1/(1 + exp(-c*z));
}

/*	A simple output routine, for ease of reuse of the net.	*/
string ANN:: netToCode()
{
	stringstream parsed;
	parsed << "/* Remember " << endl;
	parsed << "   typedef vector<double> row; " << endl;
	parsed << "   typedef vector<row> weight_matrix; */" << endl;
	parsed << endl;
	parsed << "int nin      = " << nin	<< ";" << endl;
	parsed << "int nhidd    = " << nhidd	<< ";" << endl;
	parsed << "int nout     = " << nout	<< ";" << endl;
	parsed << "weight_matrix wHO(nout);" << endl;
	parsed << "weight_matrix wIH(nhidd);" << endl;
	parsed << endl;
	for(int o_n = 0; o_n < nout; o_n++){
		for(int h_i = 0; h_i < nhidd+1; h_i++){
			parsed << "wHO[" << o_n << "][" << h_i << "] = " 
				<< wHO[o_n][h_i] << ";" << endl;
		}
	}
	parsed << endl;
	for(int h_n = 0; h_n < nhidd; h_n++){
		for(int i_i = 0; i_i < nin+1; i_i++){
			parsed << "wIH[" << h_n << "][" << i_i << "] = " 
				<< wIH[h_n][i_i] << ";" << endl;
		}
	}

	return parsed.str();	
}
