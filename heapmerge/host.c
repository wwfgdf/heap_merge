#include<dpu.h>
#include<assert.h>
#include<stdint.h>
#include<stdio.h>
#include "matrix.h"

#define NUM 10  //矩阵数量
#define ROW 10  //矩阵行数
#define APP "./app"

int main(){
    csc_matrix input_matrixs[NUM]; //输入矩阵
    int nnzs[NUM];         //每个矩阵的非零元数量
    int num_matrixs = NUM; //merge的矩阵数量
    int total_nnzs = 0;    //所有矩阵的非零元数量之和
    for(int i=0;i<NUM;i++){ //测试用初始化
        input_matrixs[i].nnz = i+1;
        input_matrixs[i].row = (int*)malloc(sizeof(int)*input_matrixs[i].nnz);
        input_matrixs[i].values = (int*)malloc(sizeof(int)*input_matrixs[i].nnz);
        for(int j=0;j<input_matrixs[i].nnz;j++){
            input_matrixs[i].row[j] = j;
            input_matrixs[i].values[j] = 1;
        }
    }

    //统计total_nnzs，便于开辟空间
    for(int i=0;i<num_matrixs;i++){
        nnzs[i] = input_matrixs[i].nnz;
        total_nnzs += nnzs[i];
    }
    struct dpu_set_t set,dpu;
    DPU_ASSERT(dpu_alloc(1, NULL, &set));
    DPU_ASSERT(dpu_load(set, APP, NULL));
    DPU_ASSERT(dpu_broadcast_to(set, "num_matrixs", 0, &num_matrixs, sizeof(int), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(set, "max_nnzs", 0, &total_nnzs, sizeof(int), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(set, DPU_MRAM_HEAP_POINTER_NAME, 0, nnzs, num_matrixs*sizeof(int), DPU_XFER_DEFAULT));
    int offset = num_matrixs*sizeof(int);
    for(int i=0;i<num_matrixs;i++){
        int len = (input_matrixs[i].nnz*sizeof(int)%8==0)?input_matrixs[i].nnz*sizeof(int):(input_matrixs[i].nnz*sizeof(int)/8*8+8);
        DPU_ASSERT(dpu_broadcast_to(set, DPU_MRAM_HEAP_POINTER_NAME, offset, input_matrixs[i].values, len, DPU_XFER_DEFAULT));
        offset += len;
    }

    for(int i=0;i<num_matrixs;i++){
        int len = (input_matrixs[i].nnz*sizeof(int)%8==0)?input_matrixs[i].nnz*sizeof(int):(input_matrixs[i].nnz*sizeof(int)/8*8+8);
        DPU_ASSERT(dpu_broadcast_to(set, DPU_MRAM_HEAP_POINTER_NAME, offset, input_matrixs[i].row, len, DPU_XFER_DEFAULT));
        offset += len;
    }

    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    int result_nnzs;
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_from(dpu, "result_nnzs", 0, &result_nnzs, sizeof(int)));
    }
    

    int* result = (int*)malloc(2*sizeof(int)*total_nnzs);
     DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, offset, result, 2*sizeof(int)*total_nnzs));
    }
   
    printf("%d\n", result_nnzs);
    printf("result values:\n");
    for(int i=0;i<result_nnzs;i++){
        printf("%d ",result[i]);
    }
    printf("\nresult rows\n");
    for(int i=0;i<result_nnzs;i++){
        printf("%d ",result[i+total_nnzs-1]);
    }
    printf("\n");
    DPU_ASSERT(dpu_free(set));
    return 0;
}