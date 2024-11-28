#pragma once

#include <iostream>


/* Basic progress bar that takes a max value and provides and increment and display method.
 * 
 * At creation time a max value is specified. Calling IncrementBar increments the internal
 * value and displays the current bar.
 *
 * IncrementBar flushes the standard output.
*/
class ProgressBar
{
private:
	int max_value = 0;
	int current_value = 0;
			
public:
	ProgressBar(int max_value) {
		this->max_value = max_value;
	}

	void IncrementBar()
	{
		std::cout << "\r" << current_value << "/" << max_value;
		std::cout.flush();
		current_value++;
	}
};
