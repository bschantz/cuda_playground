// prime.cpp : Defines the entry point for the console application.
// "Prime Sieves using Binary Quadratic Forms", Atkin, et al.
// https://cr.yp.to/papers/primesieves.pdf
//

#include "stdafx.h"



int main(void)
{
	const unsigned long MAX = 10000000;
	
	bool *result = (bool *)malloc(MAX * sizeof(bool));

	// std::vector<bool> primes;

	const unsigned long sqrt_MAX = (unsigned long)sqrt((double)MAX) + 1;

	// init all to false
	memset((void *)result, false, sizeof(bool) * MAX);

	for (int x = 1; x < sqrt_MAX; x++) {
		for (int y = 1; y < sqrt_MAX; y++) {
			// Algorithm 3.1
			int n = 4 * x * x + y * y;
			if (n <= MAX && (n % 12 == 1 || n % 12 == 5)) {
				result[n] = !result[n];
			}

			// Algorithm 3.2
			n = 3 * x * x + y * y;
			if (n <= MAX && (n % 12 == 7)) {
				result[n] = !result[n];
			}

			// Algorithm 3.3
			n = 3 * x * x - y * y;
			if (n <= MAX && x > y && n % 12 == 11) {
				result[n] = !result[n];
			}
		}
	}

	result[2] = true;
	result[3] = true;

	for (int x = 5; x < sqrt_MAX; x++) {
		if (result[x]) {
			int k = x * x;
			for (int j = k; j < MAX; j += k) {
				result[j] = false;
			}
		}
	}
	// Display prime
	long pi = 0;
	for (int i = 2; i < MAX; i++) {
		if (result[i]) {
			pi++;
			printf("%d\n", i);
		}
	}

}
