#pragma once
#include "main.h"

#define DBG_PRINT
using namespace std;

ERROR_CODE Generalized_Wiener_Attack(
	__in LINT E,
	__in LINT N,
	__out LINT *p,
	__out LINT *q,
	__out LINT *D
	)
{
#ifdef DBG_PRINT
	cout << "Starting GWA with:\nE:" << E.decstr() << endl;
	cout << "N:" << N.decstr() << endl;
#endif
	vector<LINT> P;
	vector<LINT> V;
	vector<LINT> H;
	vector<LINT> Z;

	H.push_back(E);
	Z.push_back(N);
	vector<LINT> potential_D;
	vector<LINT> potential_K;
	LINT limitD = root(root(N)) / 3 - 1;

	LINT M = LINT("10110100");
	LINT LC = mexp(M, E, N);

	//Continued fraction
	for (unsigned int i = 0;; i++)
	{
		if (Z[i] == 0) { break; }
		else
		{
			V.push_back(H[i] / Z[i]);
			Z.push_back(H[i] - Z[i] * V[i]);
			H.push_back(Z[i]);
		}
		if (i == 0)
		{
			potential_D.push_back(1);
			potential_K.push_back(V[i]);
		}
		else if (i == 1)
		{
			potential_D.push_back(V[i]);
			potential_K.push_back(V[i - 1] * V[i] + 1);
		}
		else
		{
			if ((V[i]) == 0) break;
			potential_D.push_back(potential_D[i - 1] * V[i] + potential_D[i - 2]);
			potential_K.push_back(potential_K[i - 1] * V[i] + potential_K[i - 2]);

		}
#ifdef DBG_PRINT
		cout << "Numerator : " << potential_K[i].decstr() << endl;
		cout << "Divisor : " << potential_D[i].decstr() << endl;
#endif
///Exponation
		LINT M2 = LINT(1);
		void* m2_nl = M2.n_l;
		void* lc_nl = LC.n_l;
		void* d_nl = (potential_D[i]).n_l;
		void* n_nl = N.n_l;
		size_t arraySize = (short)(*((short*)d_nl));
		cudaError_t cudaStatus = mexpWithCuda(arraySize, m2_nl, lc_nl, d_nl, n_nl);
		M2 = LINT((clint*)m2_nl);
		//LINT M2 = mexp(LC, potential_D[i], N);

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		//cudaStatus = cudaDeviceReset();
		//if (cudaStatus != cudaSuccess) {
		//	fprintf(stderr, "cudaDeviceReset failed!");
		//	return 1;
		//}
		//if (cudaStatus != cudaSuccess) {
		//	fprintf(stderr, "addWithCuda failed!");
		//	return 1;
		//}

		if (M == M2)
		{
			//Found private key
			///New part
			LINT s;
			LINT t;
			LINT p_wave;

			s = N + 1 - (E*potential_D[i]) / potential_K[i];
			t = (s*s - 4 * N);
			t = root(t);
			p_wave = (s + t) / 2;
			cout << "Result p = " << p_wave.decstr() << endl;
			*D = potential_D[i];
			*p = p_wave;
			*q = N / p_wave;

#ifdef DBG_PRINT
			cout << "Key found index " << i << endl;

			cout << "Result q = " << (N / p_wave).decstr() << endl;
#endif
			//for (int k = -3; k < 3; k++)
			//{
			//	potential_P = p_wave + (2 * k + 1)*temp_N;
			//	if (mod(N, potential_P) == LINT(0))
			//	{
			//		cout << "found P: " << potential_P.decstr() << endl;
			//		cout << "found Q: " << (N / potential_P).decstr() << endl;
			//	}
			//	////Coppersmith algorithm
			//	////
			//}
			return SUCCESS;
		}

		if (potential_D[i] > limitD)
		{
#ifdef DBG_PRINT
			cout << "poten_D: " << potential_D[i].decstr() << endl;
			cout << "limitD: " << limitD.decstr() << endl;
#endif
			*p = KEY_NOT_FOUND;
			*q = KEY_NOT_FOUND;
			cout << "E: " << E.decstr();
			cout << " N: " << N.decstr();
			return EXCEEDED_LIMIT;
		}
	}
	return LEFT_ENDLESS_LOOP;
};

int main()
{
	ERROR_CODE status = UNINITIALIZED;
	vector <LINT> primes_vector;
	LINT E;
	LINT N;
	LINT D;
	LINT P;
	LINT Q;
	LINT origin_D;
	LINT origin_P;
	LINT origin_Q;
	int key_index = 0;

#ifdef KEY_TXT_PRINT
	ofstream FILE;
	string rez = "";
	FILE.open("key for debug.txt");
#endif

	while (1 == 1)
	{
		char choice;

		cout << "\nPlease, choose what option you want:\n";
		cout << "1 - Generate vulnerable keys with specified length and break them\n";
		cout << "2 - Input prepared keys from file keys.txt and test them for being vulnerable\n";
		cout << "3 - Exit\n";

		scanf(" %c", &choice);

		switch (choice) {
		case '1':
		{
			vector<int> keys = { 1024 };//64, 128, 256, 512, 1024, 2048, 4096};

			for (unsigned int counter = 0; counter < keys.size(); counter++)
			{
				status = Prime_Number_Generator(keys[counter], &P, &Q);
				if (status != SUCCESS)
					_CrtDbgBreak();
				origin_P = P;
				origin_Q = Q;
				status = Vulnerable_Generator(P, Q, &E, &N, &origin_D);
				if (status != SUCCESS)
					_CrtDbgBreak();

				int time = GetTickCount();
				status = Generalized_Wiener_Attack(E, N, &P, &Q, &D);
				if (status != SUCCESS)
					_CrtDbgBreak();
#ifdef DBG_PRINT
				cout << "The key is: " << D.decstr() << endl;
				cout << "P is " << P.decstr() << endl;
				cout << "Q is " << Q.decstr() << endl;
#endif
				cout << "\norigin P: " << origin_P.decstr() << endl;
				if (origin_D == D)
					cout << "For key length " << keys[counter] << " Vinere succedeed in " << GetTickCount() - time << " ticks" << endl;
				if ((origin_P == P) && (origin_Q == Q))
					cout << "Primes found correctly" << endl;

#ifdef KEY_TXT_PRINT				
				rez += E.decstr();
				rez += "\n";
				rez += N.decstr();
				rez += "\n";
				rez += D.decstr();
				rez += "\n\n";
				FILE << rez;
#endif
			}

#ifdef KEY_TXT_PRINT
			FILE.close();
#endif
		}
			break;
		case '2':
		{
			string line_E;
			string line_N;
			ifstream myfile("keys.txt");
			LINT lint_E;
			LINT lint_N;
			LINT output_D;

			if (!myfile.is_open())
			{
				cout << "Couldn't open file, error." << endl;
				goto clean0;
			}

			while (!myfile.eof())
			{
				output_D = KEY_NOT_FOUND;
				getline(myfile, line_E);
				lint_E = LINT(line_E.c_str());
				getline(myfile, line_N);
				lint_N = LINT(line_N.c_str());


#ifdef DBG_PRINT
				cout << "Starting Vinere attack with E: " << lint_E << endl;
				cout << " and N: " << lint_N << endl;
#endif
				int time = GetTickCount();
				status = Generalized_Wiener_Attack(lint_E, lint_N, &P, &Q, &output_D);
				if (status != SUCCESS)
					_CrtDbgBreak();

				if (output_D != KEY_NOT_FOUND)
				{
					cout << "The key is: " << output_D.decstr() << endl;
					cout << "For key length " << (line_E.length() + line_N.length()) << " Vinere succedeed in " << GetTickCount() - time << " ticks" << endl;
					cout << "P is " << P.decstr() << endl;
					cout << "Q is " << Q.decstr() << endl;
				}
				else
				{
					cout << "The key is not vulnerable to Vinere attack. Moving to next key." << endl;
				}
			}
		}
			break;
		case '3':
		{
			goto clean0;
		}
			break;
		default:
			cout << "Invalid choice specified. Re-enter your choice" << endl;
		}
	}

clean0:
	return 0;
}