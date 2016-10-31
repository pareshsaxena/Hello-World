/* 
 Main function
*/

#include "mmtsne.h"

int main() {
	MMTSNE *mmtsne = new MMTSNE(30);
	
	mmtsne->load_input_probability_csv("association1000_P.csv", ',');
	mmtsne->construct_maps(2, 5, 400, true);
	mmtsne->save_output_vectors_csv("output1000.csv", ',');
	
	delete mmtsne;
}