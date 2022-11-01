#include<mram.h>
#include<seqread.h>
#include<alloc.h>
#include<stdio.h>
#include "matrix.h"

typedef struct{
    int value;
    int row;
    int n_matrix;
}heap_node;

void swap_heap(heap_node* a, heap_node* b){
    heap_node temp = *a;
    *a = *b;
    *b = temp;
}

void exchange_from_bottom(heap_node* heap, int pos){
    pos--;
    while(pos>0&&heap[pos].row<heap[(pos-1)/2].row){
        swap_heap(&heap[pos], &heap[(pos-1)/2]);
        pos = (pos-1)/2;
    }
}

void exchange_from_top(heap_node* heap, int pos){
    int p=0;
    int child=2*p+1;
    while(child<pos){
        if(child+1<pos&&heap[child].row>heap[child+1].row){
            child++;
        }
        if(heap[child].row<heap[p].row){
            swap_heap(&heap[child], &heap[p]);
            p = child;
            child = 2*p+1;
        }
        else{
            break;
        }
    }
}

__host int num_matrixs;  //__host
__host int max_nnzs;
__host int result_nnzs = 0;
int main(){
    int* nnzs = (int*)mem_alloc(sizeof(int)*num_matrixs);
    mram_read((__mram_ptr void const*)(DPU_MRAM_HEAP_POINTER), nnzs, sizeof(int)*num_matrixs);
    int offset = num_matrixs*sizeof(int);
    
    int** values = mem_alloc(sizeof(int*)*num_matrixs);
    int** rows = mem_alloc(sizeof(int)*num_matrixs);

    //初始化seqreader
    seqreader_buffer_t* value_cache = (seqreader_buffer_t*)mem_alloc(sizeof(seqreader_buffer_t)*num_matrixs); 
    seqreader_t* value_sr = (seqreader_t*)mem_alloc(sizeof(seqreader_t)*num_matrixs);
    for(int i=0;i<num_matrixs;i++){
        value_cache[i] = seqread_alloc();
        values[i] = seqread_init(value_cache[i], (__mram_ptr int*)(DPU_MRAM_HEAP_POINTER + offset), &value_sr[i]);
        int len = (nnzs[i]*sizeof(int)%8==0)?nnzs[i]*sizeof(int):(nnzs[i]*sizeof(int)/8*8+8);
        offset += len;
    }

    seqreader_buffer_t* row_cache = (seqreader_buffer_t*)mem_alloc(sizeof(seqreader_buffer_t)*num_matrixs); 
    seqreader_t* row_sr = (seqreader_t*)mem_alloc(sizeof(seqreader_t)*num_matrixs);
    for(int i=0;i<num_matrixs;i++){
        row_cache[i] = seqread_alloc();
        rows[i] = seqread_init(row_cache[i], (__mram_ptr int*)(DPU_MRAM_HEAP_POINTER + offset), &row_sr[i]);
        int len = (nnzs[i]*sizeof(int)%8==0)?nnzs[i]*sizeof(int):(nnzs[i]*sizeof(int)/8*8+8);
        offset += len;
    }

    heap_node* heap = (heap_node*)mem_alloc(sizeof(heap_node)*num_matrixs);
    int pos = 0;
    //每个矩阵的第一个非零元入堆
    for(int i=0;i<num_matrixs;i++){
        heap[pos].n_matrix = i;
        heap[pos].row = *(rows[i]);
        heap[pos].value = *(values[i]);
        pos++;
        rows[i] = seqread_get(rows[i], sizeof(int), &row_sr[i]);
        values[i] = seqread_get(values[i], sizeof(int), &value_sr[i]);
        nnzs[i]--;
        exchange_from_bottom(heap, pos); //向上调整堆
    }
    int* result_values = (int*)(DPU_MRAM_HEAP_POINTER + offset);
    int* result_rows = (int*)(DPU_MRAM_HEAP_POINTER + offset + sizeof(int)*max_nnzs);
    __dma_aligned int cache_values[2]; //缓存merge结果
    __dma_aligned int cache_rows[2];
    int write_index = 0;
    int prev_row = -1;
    while(pos>0){
        if(prev_row!=heap[0].row){
            if(write_index==2){
                mram_write(cache_values, (__mram_ptr void*)(result_values), 8);
                mram_write(cache_rows, (__mram_ptr void*)(result_rows), 8);
                result_values += 2;
                result_rows += 2;
                write_index = 0;
            }
            cache_values[write_index] = heap[0].value;
            cache_rows[write_index] = heap[0].row;
            write_index++;
            prev_row = heap[0].row;
            result_nnzs++;
        }
        else{
            cache_values[write_index-1] += heap[0].value;
        }
        pos--;
        int p = heap[0].n_matrix;
        swap_heap(&heap[0], &heap[pos]);
        exchange_from_top(heap, pos);//向下调整堆
        if(nnzs[p]>0){
            heap[pos].n_matrix = p;
            heap[pos].row = *(rows[p]);
            heap[pos].value = *(values[p]);
            rows[p] = seqread_get(rows[p], sizeof(int), &row_sr[p]);
            values[p] = seqread_get(values[p], sizeof(int), &value_sr[p]);
            nnzs[p]--;
            pos++;
            exchange_from_bottom(heap, pos);
        }
    }
    if(write_index>0){
        mram_write(cache_values, (__mram_ptr void*)(result_values), 8);
        mram_write(cache_rows, (__mram_ptr void*)(result_rows), 8);
        result_values += 2;
        result_rows += 2;
        write_index = 0;
    }
    return 0;
}