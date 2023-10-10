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

// Global variable for the random state used by the GMP random number generation functions.
gmp_randstate_t state;

// Function to initialize the global random state variable using the Mersenne
// Twister algorithm and current time as seed.
void initRandState() {
    gmp_randinit_mt(state);
    gmp_randseed_ui(state, std::time(0));
}

// Function to generate a random integer with a specified number of digits ('d').
void randInt(mpz_t randomInt, int d) {
    mpz_t upperLimit;
    mpz_init(upperLimit);

    // Calculate upper limit (10^d) for random number generation.
    mpz_ui_pow_ui(upperLimit, 10, d);

    // Generate a random integer in the range [0, 10^d).
    mpz_urandomm(randomInt, state, upperLimit);
    mpz_clear(upperLimit); // Clear allocated memory for upperLimit variable.
}

// Function to evaluate a polynomial using the brute force method.
void evalPolyBrute(mpz_t result, const std::vector<mpz_t>& coefficients, mpz_t x) {
    mpz_set_ui(result, 0); // Initialize result to zero.

    mpz_t term, x_power;
    mpz_init(term);
    mpz_init_set_ui(x_power, 1); // Initialize x_power to x^0 = 1

    // Iterate through each coefficient.
    for (size_t i = 0; i < coefficients.size(); ++i) {
        mpz_mul(term, x_power, coefficients[i]); // term = x_power * coefficient[i]
        mpz_add(result, result, term); // Add term to result.

        mpz_mul(x_power, x_power, x); // Update x_power for next iteration.
    }

    // Clear allocated memory for term and x_power variables.
    mpz_clear(term);
    mpz_clear(x_power);
}

// Function to evaluate a polynomial using Horner's Rule.
void evalPolyHorner(mpz_t result, const std::vector<mpz_t>& coefficients, mpz_t x) {
    mpz_set_ui(result, 0); // Initialize result to zero.

    // Iterate through coefficients in reverse order and apply Horner's Rule.
    for (int i = coefficients.size() - 1; i >= 0; --i) {
        mpz_mul(result, result, x); // Multiply current result by x.
        mpz_add(result, result, coefficients[i]); // Add current coefficient.
    }
}

// Function to print a polynomial expression.
void printPoly(const std::vector<mpz_t>& coefficients, mpz_t x) {
    std::cout << "Polynomial: P(x) = ";
    // Iterate through coefficients in reverse order to print in descending power of x.
    for (int i = coefficients.size() - 1; i >= 0; --i) {
        if (mpz_sgn(coefficients[i]) != 0) {
            if (i == 0) {
                // Print coefficient if it's the constant term.
                std::cout << mpz_get_str(nullptr, 10, coefficients[i]);
            } else {
                // Print coefficient and corresponding power of x.
                std::cout << mpz_get_str(nullptr, 10, coefficients[i]) << "x^" << i;
                if (i > 0) {
                    std::cout << " + ";
                }
            }
        }
    }
    // Print the value of x used for the polynomial evaluation.
    std::cout << " where x = " << mpz_get_str(nullptr, 10, x) << std::endl;
}

// Function to generate and initialize coefficients for a polynomial of
// degree 'n' with 'd' digits each.
std::vector<mpz_t> generateCoefficients(int n, int d) {
    std::vector<mpz_t> coefficients(n + 1);
    for (int i = 0; i <= n; ++i) {
        mpz_init(coefficients[i]);
        randInt(coefficients[i], d); // Generate random coefficient with 'd' digits.
    }
    return coefficients;
}

// Function to evaluate and benchmark a polynomial using both brute-force and
// Horner's methods, and print results.
void benchmarkAndEvaluate(const std::vector<mpz_t>& coefficients, mpz_t x, const char* size) {
    mpz_t resultBruteForce, resultHorner;
    mpz_init(resultBruteForce);
    mpz_init(resultHorner);

    // Start timing for brute-force method.
    auto startBruteForce = std::chrono::high_resolution_clock::now();
    evalPolyBrute(resultBruteForce, coefficients, x);
    auto endBruteForce = std::chrono::high_resolution_clock::now();

    // Start timing for Horner's Rule method.
    auto startHorner = std::chrono::high_resolution_clock::now();
    evalPolyHorner(resultHorner, coefficients, x);
    auto endHorner = std::chrono::high_resolution_clock::now();

    // Choose appropriate timing unit based on input size and print results.
    if (strcmp(size, "large input") == 0) {
        auto durationBruteForce = std::chrono::duration_cast<std::chrono::milliseconds>(endBruteForce - startBruteForce);
        auto durationHorner = std::chrono::duration_cast<std::chrono::milliseconds>(endHorner - startHorner);
        std::cout << "Time for Brute Force method (" << size << "): " << durationBruteForce.count() << " milliseconds\n";
        std::cout << "Time for Horner's Rule (" << size << "): " << durationHorner.count() << " milliseconds\n";
    } else {
        auto durationBruteForce = std::chrono::duration_cast<std::chrono::microseconds>(endBruteForce - startBruteForce);
        auto durationHorner = std::chrono::duration_cast<std::chrono::microseconds>(endHorner - startHorner);
        std::cout << "Time for Brute Force method (" << size << "): " << durationBruteForce.count() << " microseconds\n";
        std::cout << "Time for Horner's Rule (" << size << "): " << durationHorner.count() << " microseconds\n";

        // Print the results and elapsed times
        gmp_printf("Result (Brute Force): %Zd\n", resultBruteForce);
        gmp_printf("Result (Horner's Rule): %Zd\n", resultHorner);
    }

    // Compare results of both methods and print a message indicating whether they match.
    int comparison = mpz_cmp(resultBruteForce, resultHorner);
    if (comparison == 0) {
        std::cout << "Results (" << size << ") match.\n";
    } else {
        std::cout << "Results (" << size << ") do not match.\n";
    }

    // Clear allocated memory for result variables.
    mpz_clear(resultBruteForce);
    mpz_clear(resultHorner);
}

void clearPolyData(std::vector<mpz_t>& coefficients, mpz_t& x) {
    // Iterate over each coefficient in the vector and clear the allocated memory.
    for (size_t i = 0; i < coefficients.size(); ++i) {
        mpz_clear(coefficients[i]);
    }
    // Clear the allocated memory for the x variable.
    mpz_clear(x);
}

void processPoly(int n, int d, const char* size) {
    mpz_t x;
    mpz_init(x);
    randInt(x, d);  // Generating random integer 'x'

    auto coefficients = generateCoefficients(n, d); // Generating coefficients
    //printPoly(coefficients, x);  // Printing polynomial
    benchmarkAndEvaluate(coefficients, x, size);  // Benchmarking and evaluating polynomial

    clearPolyData(coefficients, x);  // Clearing polynomial data
}

int main() {
    initRandState();  // Initialize random state

    // processPoly(int n, int d, const char* size)
    processPoly(16, 16, "small input");  // Process polynomial for small input
    processPoly(1000, 2000, "large input");  // Process polynomial for large input

    gmp_randclear(state);  // Clear global random state variable

    return 0;
}
