#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "float.h"
#include <algorithm>
#include <typeinfo>
#include <iostream>
#include <map>
#include <vector>

#define SEGMENT (4096ul)

#define BATCH (1000)

// SCALE and N must have the following relationship: N = 2 ^ SCALE
#define SCALE (12)
#define N (4096ul)
#define DEGREE (3)
#define NNZ (DEGREE * N)

using namespace std;

typedef unsigned long spMV_data;
typedef float spMV_float;

typedef struct pack_in {
    spMV_data row;
    spMV_data col;
    spMV_float V;
} pack_in;

pack_in* dt_A = (pack_in*)malloc(NNZ * sizeof(pack_in));

bool mySort(spMV_data a, spMV_data b) {
    return(a < b);
}

int main(int argc, char* argv[]) {
    spMV_data i, j;
    printf("The number of expected non-zero elements in the whole matrix is: %lu\n", NNZ);

    float a = 0.57;
    float b = 0.19;
    float c = 0.19;

    float ab = a + b;
    float c_norm = c / (1 - ab);
    float a_norm = a / ab;

    spMV_data ii_bit = 0;
    spMV_data jj_bit = 0;
    spMV_data start_node = 0;
    spMV_data end_node = 0;

    std::map<spMV_data, std::vector<spMV_data> > nodes;

    for (int i = 0; i < NNZ; i++) {
        start_node = 0;
        end_node = 0;
        for (int ib = 0; ib < SCALE; ib++) {
            ii_bit = ((double) rand() / (RAND_MAX + 1.0)) > ab;
            jj_bit = ((double) rand() / (RAND_MAX + 1.0)) > (c_norm * ii_bit + a_norm * !ii_bit);
            start_node += pow(2, ib) * ii_bit;
            end_node += pow(2, ib) * jj_bit;
        }
        
        if (end_node < SEGMENT) {
            if (nodes.find(start_node) == nodes.end()) {
                nodes.insert(make_pair(start_node, std::vector<spMV_data>()));
            }
            if (find(nodes[start_node].begin(), nodes[start_node].end(), end_node) == nodes[start_node].end()) {
                nodes[start_node].push_back(end_node);
            }
        }
    }

    spMV_data col_iter = 0;
    for (auto& node : nodes) {
        sort(node.second.begin(), node.second.end(), mySort);
        for (int i = 0; i < node.second.size(); i++) {
            dt_A[col_iter].col = node.second[i];
            dt_A[col_iter].row = node.first;
            col_iter++;
        }
    }

    for (i = 0; i < col_iter; i++) {
        dt_A[i].V = (spMV_float) rand() / (RAND_MAX + 1.0);
    }

	printf("The number of actual non-zero elements in the block is: %u\n", col_iter);
}