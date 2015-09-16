#pragma once
#include "stdafx.h"

using namespace std;

//ERROR_CODE
#define ERROR_CODE int
#define SUCCESS 0x01
#define ZERO_COEFF 0x02
#define FAIL 0x03
#define LEFT_ENDLESS_LOOP 0x04
#define EXCEEDED_LIMIT 0x05
#define UNINITIALIZED 0xFF

ERROR_CODE extended_euclid(LINT a, LINT b, LINT *x, LINT *y, LINT *d);
ERROR_CODE Prime_Number_Generator(int length, LINT *P, LINT *Q);
ERROR_CODE Vulnerable_Generator(LINT p, LINT q, LINT *E, LINT *N, LINT *origin_D);