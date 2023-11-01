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
void evalPolyBrute(mpz_t result, // Resulting mpz_t value where the evaluation will be stored
                   const std::vector < mpz_t > & coefficients, // The polynomial coefficients
                   mpz_t x, // The x value at which polynomial is evaluated
                   size_t start = 0, // Starting index from which the polynomial should be evaluated. Default is 0 (beginning of coefficients)
                   size_t end = SIZE_MAX) { // Ending index for polynomial evaluation. Default is the max size of the system (SIZE_MAX)

  // If the end index is not specified (i.e., it's the default SIZE_MAX), set it to the size of coefficients
  if (end == SIZE_MAX) {
    end = coefficients.size();
  }

  // Initialize the result to 0
  mpz_set_ui(result, 0);

  // Declare two mpz_t variables: term for holding individual polynomial term results, and x_power for holding power of x
  mpz_t term, x_power;

  // Initialize the term variable
  mpz_init(term);

  // Initialize x_power to 1 (because anything to the power of 0 is 1)
  mpz_init_set_ui(x_power, 1);

  // If starting index is not 0, compute x to the power of starting index and store in x_power
  if (start != 0) {
    mpz_pow_ui(x_power, x, start);
  }

  // Loop from starting index to ending index to compute the polynomial value
  for (size_t i = start; i < end; ++i) {

    // Multiply current power of x with the corresponding coefficient to compute the term value
    mpz_mul(term, x_power, coefficients[i]);

    // Add the computed term value to the overall result
    mpz_add(result, result, term);

    // Multiply x_power with x to increase the power by 1 for next iteration
    mpz_mul(x_power, x_power, x);
  }

  // Clear the memory used by the term and x_power variables to avoid memory leaks
  mpz_clear(term);
  mpz_clear(x_power);
}

// Multi-threaded function to evaluate a polynomial using a brute force approach
void evalPolyBruteMT(mpz_t result, // Resulting mpz_t value where the final evaluation will be stored
                     const std::vector < mpz_t > & coefficients, // The polynomial coefficients
                     mpz_t x) { // The x value at which polynomial is evaluated

  mpz_set_ui(result, 0); // Initialize the result to 0

  // Get the number of available hardware threads on the machine
  const size_t numThreads = std::thread::hardware_concurrency();

  std::vector < std::thread > threads(numThreads); // Create a vector to hold the threads

  // Create a vector to store the local results of each thread
  std::vector < mpz_t > localResults(numThreads);

  // Calculate how many polynomial terms each thread should handle
  size_t termsPerThread = coefficients.size() / numThreads;

  for (size_t t = 0; t < numThreads; ++t) { // Launch the threads

    // Initialize the local result for the current thread
    mpz_init(localResults[t]);

    // Determine the start and end indices for the current thread
    size_t start = t * termsPerThread;
    size_t end = (t == numThreads - 1) ? coefficients.size() : start + termsPerThread;

    // Start a new thread that computes the polynomial value for the determined range of coefficients
    threads[t] = std::thread(evalPolyBrute, localResults[t], std::ref(coefficients), x, start, end);
  }

  // Wait for all threads to finish their computation
  for (auto & t: threads) {
    t.join();
  }

  // Accumulate the results from all threads
  for (mpz_t & localResult: localResults) {
    mpz_add(result, result, localResult);

    // Clear the memory used by the localResult to avoid memory leaks
    mpz_clear(localResult);
  }
}

// Function to evaluate a polynomial using Horner's Rule.
// Horner's Rule breaks down the polynomial evaluation into a nested form, which requires only
// n multiplications and n additions for a polynomial of degree n.
// Function to evaluate a polynomial using the Horner's method
void evalPolyHorner(mpz_t result, // Resulting mpz_t value where the final evaluation will be stored
                    const std::vector < mpz_t > & coefficients, // The polynomial coefficients
                    mpz_t x, // The x value at which polynomial is evaluated
                    size_t start = 0, // Starting index of coefficients to consider (default to 0)
                    size_t end = SIZE_MAX) { // Ending index of coefficients to consider (default to maximum possible value)

  // If the end index is not specified (left to its default value), set it to the size of coefficients
  if (end == SIZE_MAX) {
    end = coefficients.size();
  }

  mpz_set_ui(result, 0); // Initialize the result to 0

  // Start evaluating the polynomial from the highest term down to the term specified by the 'start' index.
  // This loop represents the Horner's method of polynomial evaluation.
  for (int i = end - 1; i >= static_cast < int > (start); --i) {

    // Multiply the current result by x
    mpz_mul(result, result, x);

    // Add the coefficient of the current term
    mpz_add(result, result, coefficients[i]);
  }
}

// Function to evaluate a polynomial using the Horner's method in a multithreaded fashion
void evalPolyHornerMT(mpz_t result, // Resulting mpz_t value where the final evaluation will be stored
                      const std::vector < mpz_t > & coefficients, // The polynomial coefficients
                      mpz_t x) { // The x value at which polynomial is evaluated

  // Initialize the result to 0
  mpz_set_ui(result, 0);

  // Determine the number of threads the hardware can run concurrently
  const size_t numThreads = std::thread::hardware_concurrency();

  // Create vectors to hold thread objects and their corresponding local results
  std::vector < std::thread > threads(numThreads);
  std::vector < mpz_t > localResults(numThreads);

  // Calculate how many terms each thread will process
  size_t termsPerThread = coefficients.size() / numThreads;

  // Create and launch threads
  for (size_t t = 0; t < numThreads; ++t) {

    // Initialize the local result for the current thread
    mpz_init(localResults[t]);

    // Determine the start and end indices for the current thread
    size_t start = t * termsPerThread;
    size_t end = (t == numThreads - 1) ? coefficients.size() : start + termsPerThread;

    // Launch the thread to evaluate a chunk of the polynomial using Horner's method
    threads[t] = std::thread(evalPolyHorner, localResults[t], std::ref(coefficients), x, start, end);
  }

  // Wait for all threads to finish
  for (auto & t: threads) {
    t.join();
  }

  // Initialize variables to accumulate results from each thread
  mpz_t powerOfX, temp;
  mpz_init(powerOfX);
  mpz_init(temp);
  mpz_set_ui(powerOfX, 1);

  // Combine results from all threads
  for (size_t t = 0; t < numThreads; ++t) {

    // Raise x to the appropriate power based on the thread's processed terms
    mpz_pow_ui(temp, x, termsPerThread * t);

    // Multiply the thread's result by the power of x to correctly position it in the polynomial
    mpz_mul(temp, localResults[t], temp);

    // Add this to the global result
    mpz_add(result, result, temp);

    // Clear the local result for the thread
    mpz_clear(localResults[t]);
  }

  // Clear temporary variables
  mpz_clear(powerOfX);
  mpz_clear(temp);
}

// Function to print a polynomial expression.
void printPoly(const std::vector < mpz_t > & coefficients, mpz_t x) {
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
std::vector < mpz_t > generateCoefficients(int n, int d) {
  std::vector < mpz_t > coefficients(n + 1);
  for (int i = 0; i <= n; ++i) {
    mpz_init(coefficients[i]);
    randInt(coefficients[i], d); // Generate random coefficient with 'd' digits.
  }
  return coefficients;
}

// Function to evaluate and benchmark a polynomial using both brute-force and
// Horner's methods, and print results.
void benchmarkAndEvaluate(const std::vector < mpz_t > & coefficients, mpz_t x,
  const char * size) {
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
    auto durationBruteForce = std::chrono::duration_cast < std::chrono::milliseconds > (endBruteForce - startBruteForce);
    auto durationBruteForceMT = std::chrono::duration_cast < std::chrono::milliseconds > (endBruteForceMT - startBruteForceMT);
    auto durationHorner = std::chrono::duration_cast < std::chrono::milliseconds > (endHorner - startHorner);
    auto durationHornerMT = std::chrono::duration_cast < std::chrono::milliseconds > (endHornerMT - startHornerMT);
    std::cout << "Time for Brute Force method (" << size << "): " << durationBruteForce.count() << " milliseconds\n";
    std::cout << "Time for Brute Force multithreaded method (" << size << "): " << durationBruteForceMT.count() << " milliseconds\n";
    std::cout << "Time for Horner's Rule (" << size << "): " << durationHorner.count() << " milliseconds\n";
    std::cout << "Time for Horner's Rule(multithreaded) (" << size << "): " << durationHornerMT.count() << " milliseconds\n";
  } else {
    auto durationBruteForce = std::chrono::duration_cast < std::chrono::microseconds > (endBruteForce - startBruteForce);
    auto durationBruteForceMT = std::chrono::duration_cast < std::chrono::microseconds > (endBruteForceMT - startBruteForceMT);
    auto durationHorner = std::chrono::duration_cast < std::chrono::microseconds > (endHorner - startHorner);
    auto durationHornerMT = std::chrono::duration_cast < std::chrono::microseconds > (endHornerMT - startHornerMT);
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

void clearPolyData(std::vector < mpz_t > & coefficients, mpz_t & x) {
  // Iterate over each coefficient in the vector and clear the allocated memory.
  for (size_t i = 0; i < coefficients.size(); ++i) {
    mpz_clear(coefficients[i]);
  }
  // Clear the allocated memory for the x variable.
  mpz_clear(x);
}

void processPoly(int n, int d,
  const char * size) {
  mpz_t x;
  mpz_init(x);
  randInt(x, d); // Generating random integer 'x'

  auto coefficients = generateCoefficients(n, d); // Generating coefficients
  // This can produces long output to screen when polynomials are long
  //printPoly(coefficients, x);  // Printing polynomial
  benchmarkAndEvaluate(coefficients, x, size); // Benchmarking and evaluating polynomial

  clearPolyData(coefficients, x); // Clearing polynomial data
}

int main() {
  initRandState(); // Initialize random state

  // Syntax: processPoly(int n, int d, const char* size)
  processPoly(32, 32, "small input"); // Process polynomial for small input
  processPoly(5000, 1000, "large input"); // Process polynomial for large input

  gmp_randclear(state); // Clear global random state variable

  return 0;
}