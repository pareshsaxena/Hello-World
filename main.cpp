/* 
 Main function
*/

#include "mmtsne.h"

int main() {
	MMTSNE *mmtsne = new MMTSNE();
	
	mmtsne->load_input_probability_csv("association1000_P.csv", ',');
	mmtsne->construct_maps(2, 5, 300, true);
	mmtsne->save_output_csv("output1000");
	

	delete mmtsne;

	return 0;
}
