/*
Programming language: C++
Programmers: Paul Moses
Repo: https://github.com/jopamo/polyeval

Date: 10.08.23
Name of class: CS3130

In this project, we will compare two different algorithms that are used to evaluate
polynomials. The goal is to understand the importance of the efficiency of an algorithm.
The first algorithm is the brute force method in which we evaluate polynomials in the
usual way. The second algorithm uses the Horner’s Rule to evaluate polynomials.

External files: The GNU Multiple Precision Arithmetic Library
                https://gmplib.org/
                Commonly packaged as 'gmp', ensure the header gmp.h is around as well
*/

#include <iostream>
#include <vector>
#include <gmp.h>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <cstring>

// Global random state
gmp_randstate_t state;

void initRandState() {
    gmp_randinit_mt(state); // Initialize random state using Mersenne Twister algorithm
    gmp_randseed_ui(state, std::time(0)); // Seed the random state with current time
}

// Function to generate a random integer with 'd' digits
void randInt(mpz_t randomInt, int d) {
    mpz_t upperLimit;
    mpz_init(upperLimit);
    mpz_ui_pow_ui(upperLimit, 10, d); // Initialize upper limit for random integer generation
    mpz_urandomm(randomInt, state, upperLimit); // Generate a random integer within the upper limit
    mpz_clear(upperLimit); // Clear the upper limit
}

// Function to evaluate a polynomial using the brute force method
void evalPolyBrute(mpz_t result, const std::vector<mpz_t>& coefficients, mpz_t x) {
    mpz_set_ui(result, 0);

    mpz_t term, x_power;
    mpz_init(term);
    mpz_init_set_ui(x_power, 1); // Initialize x_power to x^0 = 1

    for (size_t i = 0; i < coefficients.size(); ++i) {
        mpz_mul(term, x_power, coefficients[i]); // Multiply term by the corresponding coefficient
        mpz_add(result, result, term); // Add term to the result

        mpz_mul(x_power, x_power, x); // Calculate next power of x iteratively
    }

    mpz_clear(term);
    mpz_clear(x_power);
}

// Function to evaluate a polynomial using Horner's Rule
void evalPolyHorner(mpz_t result, const std::vector<mpz_t>& coefficients, mpz_t x) {
    mpz_set_ui(result, 0);

    // Iterate through coefficients vector in reverse order
    for (int i = coefficients.size() - 1; i >= 0; --i) {
        mpz_mul(result, result, x); // result = result * x
        mpz_add(result, result, coefficients[i]); // result = result + coefficients[i]
    }
}

// Function to print a polynomial expression
void printPoly(const std::vector<mpz_t>& coefficients, mpz_t x) {
    std::cout << "Polynomial: P(x) = ";
    for (int i = coefficients.size() - 1; i >= 0; --i) {
        if (mpz_sgn(coefficients[i]) != 0) {
            if (i == 0) {
                std::cout << mpz_get_str(nullptr, 10, coefficients[i]); // Print the coefficient as a string
            } else {
                std::cout << mpz_get_str(nullptr, 10, coefficients[i]) << "x^" << i; // Print the coefficient and x^i
                if (i > 0) {
                    std::cout << " + "; // Add a plus sign if not the last term
                }
            }
        }
    }
    std::cout << " where x = " << mpz_get_str(nullptr, 10, x) << std::endl; // Print the value of x
}

std::vector<mpz_t> generateCoefficients(int n, int d) {
    std::vector<mpz_t> coefficients(n + 1);
    for (int i = 0; i <= n; ++i) {
        mpz_init(coefficients[i]);
        randInt(coefficients[i], d);
    }
    return coefficients;
}

void benchmarkAndEvaluate(const std::vector<mpz_t>& coefficients, mpz_t x, const char* size) {
    mpz_t resultBruteForce, resultHorner;
    mpz_init(resultBruteForce);
    mpz_init(resultHorner);

    auto startBruteForce = std::chrono::high_resolution_clock::now();
    evalPolyBrute(resultBruteForce, coefficients, x);
    auto endBruteForce = std::chrono::high_resolution_clock::now();

    auto startHorner = std::chrono::high_resolution_clock::now();
    evalPolyHorner(resultHorner, coefficients, x);
    auto endHorner = std::chrono::high_resolution_clock::now();

    // Choose time unit based on input size
    if (strcmp(size, "large input") == 0) {
        auto durationBruteForce = std::chrono::duration_cast<std::chrono::milliseconds>(endBruteForce - startBruteForce);
        auto durationHorner = std::chrono::duration_cast<std::chrono::milliseconds>(endHorner - startHorner);

        // Print results with milliseconds unit for large input
        std::cout << "Time for Brute Force method (" << size << "): " << durationBruteForce.count() << " milliseconds\n";
        std::cout << "Time for Horner's Rule (" << size << "): " << durationHorner.count() << " milliseconds\n";
    } else {
        auto durationBruteForce = std::chrono::duration_cast<std::chrono::microseconds>(endBruteForce - startBruteForce);
        auto durationHorner = std::chrono::duration_cast<std::chrono::microseconds>(endHorner - startHorner);

        // Print results with microseconds unit for small input
        std::cout << "Time for Brute Force method (" << size << "): " << durationBruteForce.count() << " microseconds\n";
        std::cout << "Time for Horner's Rule (" << size << "): " << durationHorner.count() << " microseconds\n";
    }

    int comparison = mpz_cmp(resultBruteForce, resultHorner);
    if (comparison == 0) {
        std::cout << "Results (" << size << ") match.\n";
    } else {
        std::cout << "Results (" << size << ") do not match.\n";
    }

    mpz_clear(resultBruteForce);
    mpz_clear(resultHorner);
}

int main() {
    // Initialize the random state for the random number generator.
    initRandState();

    // Define parameters for the polynomial with small input.
    int n1 = 32; // Degree of the polynomial (small)
    int d1 = 32; // Number of digits for coefficients and x (small)

    // Generate and initialize a random x value with 'd1' digits for small input polynomial.
    mpz_t x1;
    mpz_init(x1);
    randInt(x1, d1);

    // Generate coefficients for the polynomial with small input.
    auto coefficients1 = generateCoefficients(n1, d1);

    // Print the polynomial expression for small input.
    printPoly(coefficients1, x1);

    // Evaluate and benchmark the polynomial with small input using both methods.
    benchmarkAndEvaluate(coefficients1, x1, "small input");

    // Clear memory allocated for coefficients and x in the small input polynomial.
    for (int i = 0; i <= n1; ++i) {
        mpz_clear(coefficients1[i]);
    }
    mpz_clear(x1);

    // Define parameters for the polynomial with large input.
    int n2 = 2000; // Degree of the polynomial (large)
    int d2 = 1200;  // Number of digits for coefficients and x (large)

    // Generate and initialize a random x value with 'd2' digits for large input polynomial.
    mpz_t x2;
    mpz_init(x2);
    randInt(x2, d2);

    // Generate coefficients for the polynomial with large input.
    auto coefficients2 = generateCoefficients(n2, d2);

    // Evaluate and benchmark the polynomial with large input using both methods.
    benchmarkAndEvaluate(coefficients2, x2, "large input");

    // Clear memory allocated for coefficients and x in the large input polynomial.
    for (int i = 0; i <= n2; ++i) {
        mpz_clear(coefficients2[i]);
    }
    mpz_clear(x2);

    // Clear the random number generator state.
    gmp_randclear(state);

    return 0;
}
