#pragma once
#include "stdafx.h"
#include "utils.h"

using namespace std;


#define KEY_TXT_PRINT
#define MAXIMUM_RANDOM 100
#define KEY_NOT_FOUND "-1"

ERROR_CODE Generalized_Wiener_Attack(LINT E, LINT N, LINT *D, int *key_index);
