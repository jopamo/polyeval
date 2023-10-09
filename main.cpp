/*
Programming language: C++
Programmers: Paul Moses

Date: 08.24.23
Name of class: CS3130

In this project, we will compare two different algorithms that are used to evaluate
polynomials. The goal is to understand the importance of the efficiency of an algorithm.
The first algorithm is the brute force method in which we evaluate polynomials in the
usual way. The second algorithm uses the Horner’s Rule to evaluate polynomials.

External files: The GNU Multiple Precision Arithmetic Library https://gmplib.org/
                   commonly packaged as 'gmp', ensure the header gmp.h is around as well
*/

#include <iostream>
#include <vector>
#include <gmp.h>
#include <ctime>
#include <cstdlib>
#include <chrono>

// Function to generate a random integer with 'd' digits
void randInt(mpz_t randomInt, int d) {
    mpz_t upperLimit;
    mpz_init(upperLimit);
    mpz_ui_pow_ui(upperLimit, 10, d);

    gmp_randstate_t state;
    gmp_randinit_mt(state); // Initialize random state using Mersenne Twister algorithm
    gmp_randseed_ui(state, std::time(0)); // Seed the random state

    mpz_urandomm(randomInt, state, upperLimit);

    mpz_clear(upperLimit);
    gmp_randclear(state);
}

// Function to evaluate a polynomial using the brute force method
void evalPolyBrute(mpz_t result, const std::vector<mpz_t>& coefficients, mpz_t x) {
    mpz_set_ui(result, 0);

    for (size_t i = 0; i < coefficients.size(); ++i) {
        mpz_t term;
        mpz_init(term);

        mpz_pow_ui(term, x, i);
        mpz_mul(term, term, coefficients[i]);
        mpz_add(result, result, term);

        mpz_clear(term);
    }
}

// Function to evaluate a polynomial using Horner's Rule
void evalPolyHorner(mpz_t result, const std::vector<mpz_t>& coefficients, mpz_t x) {
    mpz_set_ui(result, 0);

    for (int i = coefficients.size() - 1; i >= 0; --i) {
        mpz_mul(result, result, x);
        mpz_add(result, result, coefficients[i]);
    }
}

// Function to print a polynomial expression
void printPoly(const std::vector<mpz_t>& coefficients, mpz_t x) {
    std::cout << "Polynomial: P(x) = ";
    for (int i = coefficients.size() - 1; i >= 0; --i) {
        if (mpz_sgn(coefficients[i]) != 0) {
            if (i == 0) {
                std::cout << mpz_get_str(nullptr, 10, coefficients[i]);
            } else {
                std::cout << mpz_get_str(nullptr, 10, coefficients[i]) << "x^" << i;
                if (i > 0) {
                    std::cout << " + ";
                }
            }
        }
    }
    std::cout << " where x = " << mpz_get_str(nullptr, 10, x) << std::endl;
}

int main() {
    // First round of computation (small input)
    int n1 = 40; // Degree of the polynomial (small)
    int d1 = 4;  // Number of digits for coefficients and x (small)

    // Generate random coefficients for the polynomial
    std::vector<mpz_t> coefficients1(n1 + 1);
    for (int i = 0; i <= n1; ++i) {
        mpz_init(coefficients1[i]);
        randInt(coefficients1[i], d1);
    }

    // Generate a random x with 'd1' digits
    mpz_t x1;
    mpz_init(x1);
    randInt(x1, d1);

    // Print the polynomial expression
    //printPoly(coefficients1, x1);

    // Initialize result variables
    mpz_t resultBruteForce1;
    mpz_t resultHorner1;
    mpz_init(resultBruteForce1);
    mpz_init(resultHorner1);

    // Measure time for brute force method (small input)
    auto startBruteForce1 = std::chrono::high_resolution_clock::now();
    evalPolyBrute(resultBruteForce1, coefficients1, x1);
    auto endBruteForce1 = std::chrono::high_resolution_clock::now();

    // Measure time for Horner's Rule (small input)
    auto startHorner1 = std::chrono::high_resolution_clock::now();
    evalPolyHorner(resultHorner1, coefficients1, x1);
    auto endHorner1 = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time in microseconds (small input)
    auto durationBruteForce1 = std::chrono::duration_cast<std::chrono::microseconds>(endBruteForce1 - startBruteForce1);
    auto durationHorner1 = std::chrono::duration_cast<std::chrono::microseconds>(endHorner1 - startHorner1);

    // Compare the results (small input)
    int comparison1 = mpz_cmp(resultBruteForce1, resultHorner1);
    if (comparison1 == 0) {
        std::cout << "Results (small input) match.\n";
    } else {
        std::cout << "Results (small input) do not match.\n";
    }

    // Print the results and elapsed times (small input)
    //gmp_printf("Result (Brute Force, small input): %Zd\n", resultBruteForce1);
    //gmp_printf("Result (Horner's Rule, small input): %Zd\n", resultHorner1);

    std::cout << "Time for Brute Force method (small input): " << durationBruteForce1.count() << " microseconds\n";
    std::cout << "Time for Horner's Rule (small input): " << durationHorner1.count() << " microseconds\n";

    // Clean up GMP variables (small input)
    for (int i = 0; i <= n1; ++i) {
        mpz_clear(coefficients1[i]);
    }
    mpz_clear(x1);
    mpz_clear(resultBruteForce1);
    mpz_clear(resultHorner1);

    // Second round of computation (large input)
    int n2 = 1000; // Degree of the polynomial (large)
    int d2 = 1000;  // Number of digits for coefficients and x (large)

    // Generate random coefficients for the polynomial
    std::vector<mpz_t> coefficients2(n2 + 1);
    for (int i = 0; i <= n2; ++i) {
        mpz_init(coefficients2[i]);
        randInt(coefficients2[i], d2);
    }

    // Generate a random x with 'd2' digits
    mpz_t x2;
    mpz_init(x2);
    randInt(x2, d2);

    // Print the polynomial expression
    //printPoly(coefficients2, x2);

    // Initialize result variables
    mpz_t resultBruteForce2;
    mpz_t resultHorner2;
    mpz_init(resultBruteForce2);
    mpz_init(resultHorner2);

    // Measure time for brute force method (large input)
    auto startBruteForce2 = std::chrono::high_resolution_clock::now();
    evalPolyBrute(resultBruteForce2, coefficients2, x2);
    auto endBruteForce2 = std::chrono::high_resolution_clock::now();

    // Measure time for Horner's Rule (large input)
    auto startHorner2 = std::chrono::high_resolution_clock::now();
    evalPolyHorner(resultHorner2, coefficients2, x2);
    auto endHorner2 = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time in milliseconds (large input)
    auto durationBruteForce2 = std::chrono::duration_cast<std::chrono::milliseconds>(endBruteForce2 - startBruteForce2);
    auto durationHorner2 = std::chrono::duration_cast<std::chrono::milliseconds>(endHorner2 - startHorner2);

    // Print the results and elapsed times (large input)
    //gmp_printf("Result (Brute Force, large input): %Zd\n", resultBruteForce2);
    //gmp_printf("Result (Horner's Rule, large input): %Zd\n", resultHorner2);

    std::cout << "Time for Brute Force method (large input): " << durationBruteForce2.count() << " ms\n";
    std::cout << "Time for Horner's Rule (large input): " << durationHorner2.count() << " ms\n";

    // Clean up GMP variables (large input)
    for (int i = 0; i <= n2; ++i) {
        mpz_clear(coefficients2[i]);
    }
    mpz_clear(x2);
    mpz_clear(resultBruteForce2);
    mpz_clear(resultHorner2);

    return 0;
}
