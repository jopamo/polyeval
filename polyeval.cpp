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

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <gmp.h>
#include <iostream>
#include <thread>
#include <vector>

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
// Updated version increments by multiplying by x to increase the exponential
// value. This makes it simple multiplication in each iteration intead of exponents
// and eliminates 'term' being created/cleared with each iteration. Calculating pow
// each time old way was expensive in memory and cpu cycles.
void evalPolyBrute(mpz_t result, const std::vector<mpz_t>& coefficients, mpz_t x, size_t start = 0, size_t end = SIZE_MAX) {
    if (end == SIZE_MAX) {
        end = coefficients.size();
    }

    mpz_set_ui(result, 0);

    mpz_t term, x_power;
    mpz_init(term);
    mpz_init_set_ui(x_power, 1); // Initialize x_power to x^0 = 1
    if (start != 0) {
        mpz_pow_ui(x_power, x, start); // Initialize x_power to x^start if start is non-zero
    }

    for (size_t i = start; i < end; ++i) {
        mpz_mul(term, x_power, coefficients[i]);
        mpz_add(result, result, term);

        mpz_mul(x_power, x_power, x);
    }

    mpz_clear(term);
    mpz_clear(x_power);
}

void evalPolyBruteMT(mpz_t result, const std::vector<mpz_t>& coefficients, mpz_t x) {
    mpz_set_ui(result, 0);

    const size_t numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(numThreads);
    std::vector<mpz_t> localResults(numThreads);

    size_t termsPerThread = coefficients.size() / numThreads;

    for (size_t t = 0; t < numThreads; ++t) {
        mpz_init(localResults[t]);

        size_t start = t * termsPerThread;
        size_t end = (t == numThreads - 1) ? coefficients.size() : start + termsPerThread;

        threads[t] = std::thread(evalPolyBrute, localResults[t], std::ref(coefficients), x, start, end);
    }

    for (auto& t : threads) {
        t.join();
    }

    for (mpz_t& localResult : localResults) {
        mpz_add(result, result, localResult);
        mpz_clear(localResult);
    }
}

// Function to evaluate a polynomial using Horner's Rule.
// Horner's Rule breaks down the polynomial evaluation into a nested form, which requires only
// n multiplications and n additions for a polynomial of degree n.
void evalPolyHorner(mpz_t result, const std::vector<mpz_t>& coefficients, mpz_t x, size_t start = 0, size_t end = SIZE_MAX) {
    if (end == SIZE_MAX) {
        end = coefficients.size();
    }

    mpz_set_ui(result, 0);

    for (int i = end - 1; i >= static_cast<int>(start); --i) {
        mpz_mul(result, result, x);
        mpz_add(result, result, coefficients[i]);
    }
}

void evalPolyHornerMT(mpz_t result, const std::vector<mpz_t>& coefficients, mpz_t x) {
    mpz_set_ui(result, 0);

    const size_t numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(numThreads);
    std::vector<mpz_t> localResults(numThreads);

    size_t termsPerThread = coefficients.size() / numThreads;

    for (size_t t = 0; t < numThreads; ++t) {
        mpz_init(localResults[t]);

        size_t start = t * termsPerThread;
        size_t end = (t == numThreads - 1) ? coefficients.size() : start + termsPerThread;

        threads[t] = std::thread(evalPolyHorner, localResults[t], std::ref(coefficients), x, start, end);
    }

    for (auto& t : threads) {
        t.join();
    }

    mpz_t multiplier, temp;
    mpz_init(multiplier);
    mpz_init(temp);
    mpz_set_ui(multiplier, 1);

    for (size_t t = 0; t < numThreads; ++t) {
        mpz_mul(temp, localResults[t], multiplier);
        mpz_add(result, result, temp);

        for (size_t i = 0; i < termsPerThread; ++i) {
            mpz_mul(multiplier, multiplier, x);
        }
        mpz_clear(localResults[t]);
    }

    mpz_clear(multiplier);
    mpz_clear(temp);
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
    mpz_t resultBruteForce, resultBruteForceMT, resultHorner, resultHornerMT;
    mpz_init(resultBruteForce);
    mpz_init(resultBruteForceMT);
    mpz_init(resultHorner);
    mpz_init(resultHornerMT);

    // Start timing for brute-force method.
    auto startBruteForce = std::chrono::high_resolution_clock::now();
    evalPolyBrute(resultBruteForce, coefficients, x);
    auto endBruteForce = std::chrono::high_resolution_clock::now();

    // Start timing for brute-force MT method.
    auto startBruteForceMT = std::chrono::high_resolution_clock::now();
    evalPolyBruteMT(resultBruteForceMT, coefficients, x);
    auto endBruteForceMT = std::chrono::high_resolution_clock::now();

    // Start timing for Horner's Rule method.
    auto startHorner = std::chrono::high_resolution_clock::now();
    evalPolyHorner(resultHorner, coefficients, x);
    auto endHorner = std::chrono::high_resolution_clock::now();

    // Start timing for Horner's Rule MT method.
    auto startHornerMT = std::chrono::high_resolution_clock::now();
    evalPolyHornerMT(resultHornerMT, coefficients, x);
    auto endHornerMT = std::chrono::high_resolution_clock::now();

    // Choose appropriate timing unit based on input size and print results.
    if (strcmp(size, "large input") == 0) {
        auto durationBruteForce = std::chrono::duration_cast<std::chrono::milliseconds>(endBruteForce - startBruteForce);
        auto durationBruteForceMT = std::chrono::duration_cast<std::chrono::milliseconds>(endBruteForceMT - startBruteForceMT);
        auto durationHorner = std::chrono::duration_cast<std::chrono::milliseconds>(endHorner - startHorner);
        auto durationHornerMT = std::chrono::duration_cast<std::chrono::milliseconds>(endHornerMT - startHornerMT);
        std::cout << "Time for Brute Force method (" << size << "): " << durationBruteForce.count() << " milliseconds\n";
        std::cout << "Time for Brute Force multithreaded method (" << size << "): " << durationBruteForceMT.count() << " milliseconds\n";
        std::cout << "Time for Horner's Rule (" << size << "): " << durationHorner.count() << " milliseconds\n";
        std::cout << "Time for Horner's Rule(multithreaded) (" << size << "): " << durationHornerMT.count() << " milliseconds\n";
    } else {
        auto durationBruteForce = std::chrono::duration_cast<std::chrono::microseconds>(endBruteForce - startBruteForce);
        auto durationBruteForceMT = std::chrono::duration_cast<std::chrono::microseconds>(endBruteForceMT - startBruteForceMT);
        auto durationHorner = std::chrono::duration_cast<std::chrono::microseconds>(endHorner - startHorner);
        auto durationHornerMT = std::chrono::duration_cast<std::chrono::microseconds>(endHornerMT - startHornerMT);
        std::cout << "Time for Brute Force method (" << size << "): " << durationBruteForce.count() << " microseconds\n";
        std::cout << "Time for Brute Force multithreaded method (" << size << "): " << durationBruteForceMT.count() << " microseconds\n";
        std::cout << "Time for Horner's Rule (" << size << "): " << durationHorner.count() << " microseconds\n";
        std::cout << "Time for Horner's Rule(multithreaded) (" << size << "): " << durationHornerMT.count() << " microseconds\n";

        // Print the results and elapsed times
        gmp_printf("Result (Brute Force): %Zd\n", resultBruteForce);
        gmp_printf("Result (Brute Force MT): %Zd\n", resultBruteForceMT);
        gmp_printf("Result (Horner's Rule): %Zd\n", resultHorner);
        gmp_printf("Result (Horner's Rule MT): %Zd\n", resultHornerMT);
    }

    // Compare results of both methods and print a message indicating whether they match.
    int comparison = mpz_cmp(resultBruteForce, resultHorner);
    int comparison2 = mpz_cmp(resultBruteForce, resultBruteForceMT);
    int comparison3 = mpz_cmp(resultBruteForceMT, resultHornerMT);

    if (comparison == 0 && comparison2 == 0 && comparison3 == 0) {
        std::cout << "Results (" << size << ") match.\n";
    } else {
        std::cout << "Results (" << size << ") do not match.\n";
    }

    // Clear allocated memory for result variables.
    mpz_clear(resultBruteForce);
    mpz_clear(resultHorner);
    mpz_clear(resultBruteForceMT);
    mpz_clear(resultHornerMT);
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
    processPoly(32, 32, "small input");  // Process polynomial for small input
    processPoly(10000, 1000, "large input");  // Process polynomial for large input

    gmp_randclear(state);  // Clear global random state variable

    return 0;
}