#pragma once
#include "utils.h"	

using namespace std;

/*Function extended_euclid

This function calculates coefficients a, b and GCD(a,b)
in comparison a*x+b*y=GCD(a,b)
The answer is stored three LINT: x,y,d

Arguments:
IN:
a - the first coefficient in comparison
b - the second coefficient in comparison
OUT:
x - the first variable in comparison
y - the second variable in comparison
d - greatest common divisor of a and b

Return value:
None
*/

ERROR_CODE extended_euclid(
	__in	LINT a,
	__in	LINT b,
	__out	LINT *x,
	__out	LINT *y,
	__out	LINT *d)


	/* calculates a * *x + b * *y = gcd(a, b) = *d */

{

	LINT q, r, x1, x2, y1, y2;
	if (b == 0)
	{
		*d = a, *x = 1, *y = 0;
		return ZERO_COEFF;
	}
	x2 = 1, x1 = 0, y2 = 0, y1 = 1;

	while (b > 0)
	{
		q = a / b, r = a - q * b;
		*x = x2 - q * x1, *y = y2 - q * y1;
		a = b, b = r;
		x2 = x1, x1 = *x, y2 = y1, y1 = *y;
	}
	*d = a, *x = x2, *y = y2;

	return SUCCESS;
}

/*
Produces two random prime numbers
by generating random number and taking
next nearest prime one

Arguments:
IN:
length - the number of bits in result prime numbers
OUT:
P - first number
Q - second number

Return value:
None
*/
ERROR_CODE Prime_Number_Generator(
	__in int length,
	__out LINT *P,
	__out LINT *Q)
{
	*P = nextprime(randl(length / 2) + 1, 1);
	*Q = nextprime(randl(length / 2) + 1, 1);
#ifdef DBG_PRINT
	cout << "Finished generating primes" << endl;
	cout << "P: " << (*P).decstr() << endl;
	cout << "Q: " << (*Q).decstr() << endl;
#endif
	return SUCCESS;
}

/*
Function Vulnerable_Generator

This function recieves public part of key from
pair p,q
The answer is the public part of key E,N

Arguments:

IN:
p - first prime
q - second prime
OUT:
*E - the first part of public key
*N - the second part of public key
*origin_D - private key

Return value:
None
*/

ERROR_CODE Vulnerable_Generator(
	__in  LINT p,
	__in  LINT q,
	__out LINT *E,
	__out LINT *N,
	__out LINT *origin_D)
{
	LINT NOD;
	LINT koef;

	*N = mul(p, q);
	LINT eiler_func = mul((p - 1), (q - 1));
	LINT limitD = root(root(*N)) / 3 - 1;
	for (; limitD > 0; limitD--)
	{
		extended_euclid(limitD, eiler_func, E, &koef, &NOD); // E - output

		LINT one = LINT(1);

		if ((*E < *N) && (NOD == 1) && (gcd(*E, eiler_func) == 1) && ((*E) != one))
		{
#ifdef DBG_PRINT
			cout << endl << "Eiler: " << eiler_func.decstr() << endl;
			cout << "Finished generating E, D and N." << endl;
			cout << "E: " << (*E).decstr() << endl;
			cout << "D: " << limitD.decstr() << endl;
			cout << "N: " << (*N).decstr() << endl;
			cout << "p: " << p.decstr() << endl;
			cout << "q: " << q.decstr() << endl;
#endif
			*origin_D = limitD;
			return SUCCESS;
		}
	}
	return FAIL;
}