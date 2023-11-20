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
#include <cstring>
#include <gmp.h>
#include <iostream>
#include <thread>
#include <vector>

void printUsage(const std::string& programName) {
  std::cout << "Usage: " << programName << " <Degree of a polynomial> <Length in digits> [OPTIONS]\n";
  std::cout << "Example: " << programName << " 100 200 [OPTIONS]\n\n";

  std::cout << "Options:\n";
  std::cout << "  -r, --results     Print the resulting value.\n";
  std::cout << "  -p, --print-poly  Print the generated polynomial.\n";
  std::cout << "  -x, --print-x     Print the randomly generated x value.\n";
  std::cout << "  -b, --benchmark   Run slower methods to compare results.\n\n";

  std::cout << "Description:\n";
  std::cout << "  This program evaluates a polynomial with randomly generated coefficients using\n";
  std::cout << "  various methods. The user can specify the Degree of a polynomial (n) and the length\n";
  std::cout << "  in digits (d) of each randomly generated coefficient and the random value of (x).\n";
  std::cout << "  Additional options allow for the printing of resulting value, the polynomial solved,\n";
  std::cout << "  the x value used in the evaluation, and to run the slower methods for comparison.\n";
}

// Function pointer type for polynomial evaluation methods.
typedef void (*EvalPolyFunc)(mpz_t, const std::vector<mpz_t>&, mpz_t, size_t, size_t);

bool tryStoi(const std::string& str, int& out, const std::string& programName) {
  try {
    out = std::stoi(str);
    return true;
  }
  catch (const std::out_of_range& e) {
    std::cout << "That number is way too big, 99999 max for both. This uses 4GB of RAM or so. - ./polyeval <n> <d>\n\n";
    printUsage(programName);
  }
  catch (const std::invalid_argument& e) {
    std::cout << "You must enter numbers n >= 0 and d > 0 - ./polyeval <n> <d>\n\n";
    printUsage(programName);
  }
  return false;
}

// Function to initialize and return a new random state
void initRandState(gmp_randstate_t randState) {
  gmp_randinit_mt(randState); // or another suitable initialization method
  gmp_randseed_ui(randState, static_cast<unsigned long>(std::time(nullptr)));
}

// Generates a random integer with exactly d digits, or d-1 digits if includeZero is true.
void randInt(mpz_t randomInt, gmp_randstate_t randState, int d, bool includeZero) {
  mpz_t lowerLimit, upperLimit;

  // Initialize mpz variables for the lower and upper limits.
  mpz_inits(lowerLimit, upperLimit, NULL);

  // Set the lower limit based on d and includeZero.
  if (d > 1) {
    mpz_ui_pow_ui(lowerLimit, 10, d - 1); // lowerLimit = 10^(d-1)
  }
  else { // d = 1
    if (includeZero) {
      mpz_set_ui(lowerLimit, 0); // lowerLimit = 0
    }
    else {
      mpz_set_ui(lowerLimit, 1); // lowerLimit = 1 for single-digit non-zero numbers
    }
  }

  // Set the upper limit to 10^d.
  mpz_ui_pow_ui(upperLimit, 10, d); // upperLimit = 10^d

  // Generate a random number within [0, upperLimit).
  mpz_urandomm(randomInt, randState, upperLimit);

  // If d > 1, ensure the number has d digits by adding the lower limit.
  if (d > 1) {
    mpz_add(randomInt, randomInt, lowerLimit);
  }

  // If, after adding the lower limit, the number equals or exceeds the upper limit, correct it.
  if (mpz_cmp(randomInt, upperLimit) >= 0) {
    mpz_sub(randomInt, randomInt, upperLimit);
  }

  // Clean up mpz variables to prevent memory leaks.
  mpz_clears(lowerLimit, upperLimit, NULL);
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

// Multi-threaded polynomial evaluation function
void evalPolyMT(mpz_t result, // Resulting mpz_t value
                const std::vector<mpz_t> &coefficients, // Polynomial coefficients
                mpz_t x, // The x value
                // Pointer to either evalPolyBrute or evalPolyHorner
                EvalPolyFunc evalFunc) {

  // Initialize result
  mpz_set_ui(result, 0);

  // Determine the number of threads and set up threading infrastructure
  size_t numThreads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads(numThreads);
  std::vector<mpz_t> localResults(numThreads);
  size_t termsPerThread = coefficients.size() / numThreads;

  // Determine if Horner's method is used
  bool isHorner = (evalFunc == evalPolyHorner);

  // Launch threads with the provided polynomial evaluation function
  for (size_t t = 0; t < numThreads; ++t) {
    mpz_init(localResults[t]);
    size_t start = t * termsPerThread;
    size_t end = (t == numThreads - 1) ? coefficients.size() : start + termsPerThread;
    threads[t] = std::thread(evalFunc, localResults[t], std::ref(coefficients), x, start, end);
  }

  // Initialize variables for Horner's method adjustment
  mpz_t powerOfX, temp;
  if (isHorner) {
    mpz_init(powerOfX);
    mpz_init(temp);
    mpz_set_ui(powerOfX, 1);
  }

  // Join threads and accumulate results
  for (size_t t = 0; t < numThreads; ++t) {
    threads[t].join();

    if (isHorner) {
      // Adjust for Horner's method
      mpz_pow_ui(temp, x, termsPerThread * t);
      mpz_mul(temp, localResults[t], temp);
      mpz_add(result, result, temp);
    }
    else {
      // Brute force method
      mpz_add(result, result, localResults[t]);
    }

    mpz_clear(localResults[t]);
  }

  // Clear temporary variables for Horner's method
  if (isHorner) {
    mpz_clear(powerOfX);
    mpz_clear(temp);
  }
}

// Function to print a polynomial expression.
void printPoly(const std::vector<mpz_t>& coefficients, mpz_t x) {
  std::cout << "\nPolynomial: P(x) = ";
  bool firstPrinted = false; // '+' sign before non-first terms

  // Reverse order to print in descending power of x.
  for (int i = coefficients.size() - 1; i >= 0; --i) {
    if (mpz_sgn(coefficients[i]) != 0) { //print non-zero coefficients
      // '+' sign for non-first terms
      if (firstPrinted) {
        std::cout << " + ";
      }
      firstPrinted = true;

      // Print coefficient
      std::cout << mpz_get_str(nullptr, 10, coefficients[i]);

      // Print the corresponding power of x if it's not the constant term
      if (i > 0) {
        std::cout << "x";
        if (i > 1) {
          std::cout << "^" << i;
        }
      }
    }
  }
  // Check if all coefficients were zero
  if (!firstPrinted) {
    std::cout << "0";
  }
  // Print the value of x
  std::cout << " where x = " << mpz_get_str(nullptr, 10, x) << "\n\n";
}

// Function to generate and initialize coefficients for a polynomial of
// degree 'n' with 'd' digits each.
std::vector<mpz_t> generateCoefficients(gmp_randstate_t randState, int n, int d) {
  std::vector<mpz_t> coefficients(n + 1);  // create vector with 'n + 1'
  for (int i = 0; i <= n; ++i) {            // 'n + 1'
    mpz_init(coefficients[i]);
    randInt(coefficients[i], randState, d, false);
  }
  return coefficients;
}

// Function to evaluate and benchmark a polynomial using both brute-force and
// Horner's methods, and print results.
void benchmarkAndEvaluate(const std::vector<mpz_t>& coefficients, mpz_t x, bool giveResults,  bool giveBench) {
    mpz_t resultHornerMT;
    mpz_init(resultHornerMT);

    mpz_t resultBruteForce, resultBruteForceMT, resultHorner;

    mpz_init(resultBruteForce);
    mpz_init(resultBruteForceMT);
    mpz_init(resultHorner);

    // Function to print time appropriately
    auto printTime = [](const std::chrono::microseconds& duration, const char* method) {
      if (duration.count() >= 1000) {
        std::cout << "Time for " << method << static_cast<float>(duration.count()) / 1000.0 << " milliseconds\n";
      }
      else {
        std::cout << "Time for " << method << duration.count() << " microseconds\n";
      }
    };

    if (giveBench) {
      // Start timing for brute-force method.
      auto startBruteForce = std::chrono::high_resolution_clock::now();
      evalPolyBrute(resultBruteForce, coefficients, x);
      auto endBruteForce = std::chrono::high_resolution_clock::now();
      auto durationBruteForce = std::chrono::duration_cast<std::chrono::microseconds>(endBruteForce - startBruteForce);
      printTime(durationBruteForce, "Brute Force method:\t\t\t");

      // Start timing for brute-force mt method.
      auto startBruteForceMT = std::chrono::high_resolution_clock::now();
      evalPolyMT(resultBruteForceMT, coefficients, x, evalPolyBrute);
      auto endBruteForceMT = std::chrono::high_resolution_clock::now();
      auto durationBruteForceMT = std::chrono::duration_cast<std::chrono::microseconds>(endBruteForceMT - startBruteForceMT);
      printTime(durationBruteForceMT, "Brute Force method (multithreaded):\t");

      // Start timing for horner method.
      auto startHorner = std::chrono::high_resolution_clock::now();
      evalPolyHorner(resultHorner, coefficients, x);
      auto endHorner = std::chrono::high_resolution_clock::now();
      auto durationHorner = std::chrono::duration_cast<std::chrono::microseconds>(endHorner - startHorner);
      printTime(durationHorner, "Horner's method:\t\t\t");
    }

    // Start timing for horner mt method.
    auto startHornerMT = std::chrono::high_resolution_clock::now();
    evalPolyMT(resultHornerMT, coefficients, x, evalPolyHorner);
    auto endHornerMT = std::chrono::high_resolution_clock::now();
    auto durationHornerMT = std::chrono::duration_cast<std::chrono::microseconds>(endHornerMT - startHornerMT);
    printTime(durationHornerMT, "Horner's method (multithreaded):\t");

    // Print the results
    if (giveResults) {
      if (giveBench) {
        gmp_printf("Result (Brute Force): %Zd\n", resultBruteForce);
        gmp_printf("Result (Brute Force MT): %Zd\n", resultBruteForceMT);
        gmp_printf("Result (Horner's Rule): %Zd\n", resultHorner);
      }
      gmp_printf("Result (Horner's Rule MT): %Zd\n", resultHornerMT);
    }

    if (giveBench) {
      // Compare results of both methods and print a message indicating whether they match.
      int comparison = mpz_cmp(resultBruteForce, resultHorner);
      int comparison2 = mpz_cmp(resultBruteForce, resultBruteForceMT);
      int comparison3 = mpz_cmp(resultBruteForceMT, resultHornerMT);

      if (comparison == 0 && comparison2 == 0 && comparison3 == 0) {
        std::cout << "  Results match.\n";
      }
      else {
        std::cout << "  Results do not match.\n";
      }

      // Clear allocated memory for result variables.
      mpz_clear(resultBruteForce);
      mpz_clear(resultHorner);
      mpz_clear(resultBruteForceMT);
    }
  mpz_clear(resultHornerMT);
}


void clearPolyData(std::vector < mpz_t > & coefficients, mpz_t & x) {
  // Iterate over each coefficient in the vector and clear the allocated memory.
  for (mpz_t& coeff : coefficients) {
    mpz_clear(coeff);
  }
  // Clear the allocated memory for the x variable.
  mpz_clear(x);
}

void processPoly(gmp_randstate_t randState, int n, int d, bool giveResults, bool givePoly, bool giveX, bool giveBench) {
  mpz_t x;
  mpz_init(x);
  randInt(x, randState, d, false); // Generating random integer 'x'

  if (giveX) {
    gmp_printf("Value for x: %Zd\n", x);
  }

  auto coefficients = generateCoefficients(randState, n, d); // Generating coefficients

  // This can produce long output to screen when polynomials are long
  if (givePoly) {
    printPoly(coefficients, x);  // Printing polynomial
  }

  benchmarkAndEvaluate(coefficients, x, giveResults, giveBench); // Benchmarking and evaluating polynomial

  clearPolyData(coefficients, x); // Clearing polynomial data
}

bool parseArgs(int argc, char* argv[], int& n, int& d, bool& giveResults, bool& printPoly, bool& giveX, bool& giveBench) {
  giveResults = false;
  printPoly = false;
  giveX = false;
  giveBench = false;

  if (argc < 3 || argc > 7) {
    printUsage(argv[0]);
    return false;
  }

  if (!tryStoi(argv[1], n, argv[0]) || !tryStoi(argv[2], d, argv[0])) {
    // tryStoi has already printed the error message.
    return false;
  }

  if (n < 0) {
    std::cerr << "'n' must be positive.\n";
    return false;
  }

  if (d <= 0) {
    std::cerr << "'d' needs to be greater than zero.\n";
    return false;
  }

  if (n > 99999) {
    std::cerr << "n cannot be greater than 99999\n";
    return false;
  }

  if (d > 99999) {
    std::cerr << "d cannot be greater than 99999\n";
    return false;
  }

  for (int i = 3; i < argc; i++) {
    if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--results") == 0) {
      giveResults = true;
    }
    else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--print-poly") == 0) {
      printPoly = true;
    }
    else if (strcmp(argv[i], "-x") == 0 || strcmp(argv[i], "--print-x") == 0) {
      giveX = true;
    }
    else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--benchmark") == 0) {
      giveBench = true;
    }
    else {
      std::cerr << "Invalid option: " << argv[i] << std::endl;
      printUsage(argv[0]);
      return false;
    }
  }

  return true;
}

int main(int argc, char* argv[]) {
  gmp_randstate_t randState; // Declare the random state
  initRandState(randState);  // Initialize the random state

  int n, d;
  bool giveResults, givePoly, giveX, giveBench;

  if (!parseArgs(argc, argv, n, d, giveResults, givePoly, giveX, giveBench)) {
    return 1;
  }

  std::cout << "Value for n: " << n << "\n";
  std::cout << "Value for d: " << d << "\n";

  processPoly(randState, n, d, giveResults, givePoly, giveX, giveBench);

  gmp_randclear(randState); // Clear global random state variable

  return 0;
}
