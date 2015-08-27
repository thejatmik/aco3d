#include "cuda.h"
#include "conio.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <assert.h>
#include "book.h"
//#include "cpu_anim.h"
#include <fstream>
#include <string>

using namespace std;

__global__ void kernel_p0 (int *DDIMX, int *DDIMY, int *DDIMZ, float *vx, float *vy, float *vz, float *p0, float *ddx, float *ddy, float *ddz, float *kappaloc, float *buoloc, float *value_dvx_dx, float *value_dvy_dy, float *value_dvz_dz, int *is, int *js, int *ks, int *i_gpu, float *dt, float *a, float *f0)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = blkId * aaaa + bbbb +cccc;

	int tx	= blockDim.x*gridDim.x;
	int ty	= blockDim.y*gridDim.y;

	int left	= offset-1;
	int left2   = offset-2;
	int right	= offset+1;
	int top     = offset+(DDIMX[0]);
	int bottom	= offset-(DDIMX[0]);
	int bottom2	= offset-(2*DDIMX[0]);
	int front	= offset+(DDIMX[0]*DDIMY[0]);
	int rear	= offset-(DDIMX[0]*DDIMY[0]);
	int rear2	= offset-(2*DDIMX[0]*DDIMY[0]);
	/* no cpml
	if ((index_z >= 1) && (index_z < DDIMZ[0]-1)) {
		if ((index_y >= 1) && (index_y < DDIMY[0]-1)) {
			if ((index_x >= 1) && (index_z < DDIMX[0]-1)) {
				dp0[offset]	= dp0[offset] - cfl[0]*(du0[offset]-du0[left]+dv0[offset]-dv0[bottom]+dw0[offset]-dw0[rear]);
			}
		}
	}
	*/

	if ((index_z >= 1) && (index_z < DDIMZ[0]-1)) {
		if ((index_y >= 1) && (index_y < DDIMY[0]-1)) {
			if ((index_x >= 1) && (index_x < DDIMX[0]-1)) {
				/*dvx_dx = ( a0*(vx[offset]-vx[left]) + a1*(vx[right]-vx[left2]) )/ddx[0];
				dvy_dy = ( a0*(vy[offset]-vy[bottom]) + a1*(vy[top]-vy[bottom2] )/ddy[0];
				dvz_dz = ( a0*(vz[offset]-vz[rear]) + a1*(vz[front]-vz[rear2]) )/ddz[0];*/

				value_dvx_dx = (27.0*vx[offset] - 27.0*vx[left] - vx[right] + vx[left2])/ddx[0]/24.0;
				value_dvy_dy = (27.0*vy[offset] - 27.0*vy[bottom] - vy[top] + vy[bottom2]) / ddy[0] / 24.0;
				value_dvz_dz = (27.0*vz[offset] - 27.0*vz[rear] - vz[front] + vz[rear2]) / ddz[0] / 24.0;

				p0[offset] = p0[offset] + kappaloc[offset] * (value_dvx_dx + value_dvy_dy + value_dvz_dz);
			}
		}
	}

	if ((index_x == is[0]) && (index_y == js[0]) && (index_z == ks[0])) {
		float t = i_gpu[0]*dt[0];
		float faktor = 10e4;
		//float source=- faktor * 2.0*a*(t-1.2/f0[0])*exp(-a*pow(t-1.2/f0[0],2));                                 
		//faktor * exp(-a*pow((t-0.12/f0[0]),2));
		float w = faktor * (1.0 - 2.0*a*pow(t-1.2/f0[0],2))*exp(-a*pow(t-1.2/f0[0],2));
        //tzz[offset] = tzz[offset] + source; //dt[0]*source/dz[0];
		p0[offset] = p0[offset]+w*buoloc[offset];
		i_gpu[0] = i_gpu[0] + 1.0;
		}
}

__global__ void kernel_vx (int *DDIMX, int *DDIMY, int *DDIMZ, float *vx, float *p0, float *ddx, float *buoloc, float *value_dp_dx) 
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = blkId * aaaa + bbbb +cccc;
	int right = offset+1;
	int right2 = offset+2;
	int left  = offset-1;
	int left2 = offset-2;

	/* no cpml
	if ((index_z >= 0) && (index_z < DDIMZ[0]-1)) {
		if ((index_y >= 0) && (index_y < DDIMY[0]-1)) {
			if ((index_x >= 0) && (index_z < DDIMX[0]-2)) {
				du0[offset] = du0[offset] - cfl[0]*(dp0[right]-dp0[offset]);
			}
		}
	}
	*/

	if ((index_z >= 0) && (index_z < DDIMZ[0]-1)) {
		if ((index_y >= 0) && (index_y < DDIMY[0]-1)) {
			if ((index_x >= 0) && (index_x < DDIMX[0]-2)) {
				//dp_dx = ( a0*(p0[right]-p0[offset]) + a1*(p0[right2]-p0[left]) )/ddx;
				
				value_dp_dx = (27.0*p0[right] - 27.0*p0[offset] - p0[left] + p0[right2]) / ddx[0] / 24.0;
				vx[offset] = vx[offset] + buoloc[offset]*value_dp_dx;
			}
		}
	}
}

__global__ void kernel_vy (int *DDIMX, int *DDIMY, int *DDIMZ, float *vy, float *p0, float *ddy, float *buoloc, float *value_dp_dy) 
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = blkId * aaaa + bbbb +cccc;
	int top = offset + DDIMX[0];
	int top2 = offset + (2*DDIMX[0]);
	int bottom = offset - DDIMX[0];

	/* no cpml
	if ((index_z >= 0) && (index_z < DDIMZ[0]-1)) {
		if ((index_y >= 0) && (index_y < DDIMY[0]-2)) {
			if ((index_x >= 0) && (index_z < DDIMX[0]-1)) {
				dv0[offset] = dv0[offset] - cfl[0]*(dp0[top]-dp0[offset]);
			}
		}
	}*/

	if ((index_z >= 0) && (index_z < DDIMZ[0]-1)) {
		if ((index_y >= 0) && (index_y < DDIMY[0]-2)) {
			if ((index_x >= 0) && (index_x < DDIMX[0]-1)) {
				//dp_dy = ( a0*(p0[top]-p0[offset]) + a1*(p0[top2]-p[bottom]) )/ddy;
				value_dp_dy = (27.0*p0[top] - 27.0*p0[offset] - p0[bottom] + p0[top2]) / ddy[0] / 24.0;
				vy[offset] = vy[offset] + buoloc[offset]*value_dp_dy;
			}
		}
	}
}

__global__ void kernel_vz (int *DDIMX, int *DDIMY, int *DDIMZ, float *vz, float *p0, float *ddz, float *buoloc, float *value_dp_dz) 
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = blkId * aaaa + bbbb +cccc;
	int front = offset + DDIMX[0]*DDIMY[0];
	int front2 = offset + 2*DDIMX[0]*DDIMY[0];
	int rear  = offset - DDIMX[0]*DDIMY[0];
	
	/*no cpml
	if ((index_z >= 0) && (index_z < DDIMZ[0]-2)) {
		if ((index_y >= 0) && (index_y < DDIMY[0]-1)) {
			if ((index_x >= 0) && (index_z < DDIMX[0]-1)) {
				dw0[offset] = dw0[offset]-cfl[0]*(dp0[front]-dp0[offset]);
			}
		}
	}*/

	if ((index_z >= 0) && (index_z < DDIMZ[0]-2)) {
		if ((index_y >= 0) && (index_y < DDIMY[0]-1)) {
			if ((index_x >= 0) && (index_x < DDIMX[0]-1)) {
				//dp_dz = ( a0*(p0[front]-p0[offset]) + a1*(p0[front2]-p0[rear]) )/ddz;
				value_dp_dz = (27.0*p0[front] - p0[offset] - p0[rear] + p0[front2]) / ddz[0] / 24.0;
				vz[offset] = vz[offset] + bulloc[offset]*value_dp_dz;
			}
		}
	}
}

__global__ void kernel_select_trace()
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = blkId * aaaa + bbbb + cccc;

}

__global__ void kernel_save_snap(int *DDIMX, int *DDIMY, int *DDIMZ, int *sslicex, int *sslicey, int *sslicez, float *vx, float *vy, float *vz, float *sliceisox_vx, float *sliceisox_vz, float *sliceisoy_vx, float *sliceisoy_vz, float *sliceisoz_vx, float *sliceisoz_vz)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = blkId * aaaa + bbbb + cccc;
	int ij = index_x + index_y*blockDim.x*gridDim.x;
	int ik = index_x + index_z*blockDim.x*gridDim.x;
	int jk = index_y + index_z*blockDim.x*gridDim.x;

	if (index_z == sslicez[0]) {
		if ((index_y >= 0) && (index_y < DDIMY[0] - 1)) {
			if ((index_x >= 0) && (index_z < DDIMX[0] - 1)) {
				sliceisoz_vx[ij]=vx[offsset];
				sliceisoz_vz[ij]=vz[offset];
			}
		}
	}

	if ((index_z >= 0) && (index_z < DDIMZ[0]-1)) {
		if (index_y == sslicey[0]) {
			if ((index_x >= 0) && (index_z < DDIMX[0]-1)) {
				sliceisoy_vx[ik]=vx[offset];
				sliceisoy_vz[ik]=vz[offset];
			}
		}
	}

	if ((index_z >= 0) && (index_z < DDIMZ[0]-1)) {
		if ((index_y >= 0) && (index_y < DDIMY[0]-2)) {
			if (index_x ==sslicex[0]) {
				sliceisox_vx[jk]=vx[offset];
				sliceisox_vz[jk]=vz[offset];
			}
		}
	}
}

__global__ void kernel_save_seismo () 
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = blkId * aaaa + bbbb +cccc;


}

template <typename T>
T **AllocateDynamicArray( int nRows, int nCols)
{
      T **dynamicArray;

      dynamicArray = new T*[nRows];
      for( int i = 0 ; i < nRows ; i++ )
      dynamicArray[i] = new T [nCols];

      return dynamicArray;
}

template <typename T>
void FreeDynamicArray(T** dArray)
{
      delete [] *dArray;
      delete [] dArray;
}

int main(void) {
	std::ifstream in_file1("input1.txt");
	if (!infile1.is_open())
	{
		printf("Fail to open file");
		return 1;
	}

	std::ifstream in_file2("input2.txt");
	if (!infile2.is_open())
	{
		printf("Fail to open file");
		return 1;
	}

	std::ifstream in_file3("input3.txt");
	if (!infile3.is_open())
	{
		printf("Fail to open file");
		return 1;
	}
	
	int DIMX, DIMY, DIMZ, iis, jjs, kks, nite, nsamp;
	float ddt, dx, dy, dz, ff0;
	int vsp, hor1, hor2, slicex, slicey, slicez;

	std::string line;

	//open file input 1
	in_file1 >> nite;
	in_file1 >> ddt;
	in_file1 >> nsamp;
	in_file1 >> iis;
	in_file1 >> jjs;
	in_file1 >> kks;

	//open file input 2
	in_file2 >> ff0;
	in_file2 >> DIMX;
	in_file2 >> DIMY;
	in_file2 >> DIMZ;
	in_file2 >> vsp;
	in_file2 >> hor1;
	in_file2 >> hor2;
	
	in_file3 >> slicex;
	in_file3 >> slicey;
	in_file3 >> slicez;
	
	nite	= 100;
	ddt		= 0.01;
	nsamp = 10;
	DIMX = 100;
	DIMY = 100;
	DIMZ = 100;
	iis = 50;
	jjs = 50;
	kks = 10;

	ff0 = 20.0;
	vsp = 5;
	hor1 = 5;
	hor2 = 5;

	slicex=50;
	slicey=50;
	slicez=50;

	int NSTEP = nite;
	int NX = DIMX-1;
	int NY = DIMY-1;
	int NZ = DIMZ-1;
	int xthread = 10;
	int ythread = 10;
	int zthread = 10;

	HANDLE_ERROR(cudaMalloc((void**)&DDIMX, 1*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&DDIMY, 1*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&DDIMZ, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&is, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&ks, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&js, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dx, sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dy, sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dz, sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&f0, sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dt, sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&sslicex, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&sslicey, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&sslicez, sizeof(int)));

	HANDLE_ERROR(cudaMemcpy(DDIMX, &DIMX, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(DDIMY, &DIMY, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(DDIMZ, &DIMZ, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(is, &iis, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(js, &jjs, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(ks, &kks, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(ddx, &dx, sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(ddy, &dy, sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(ddz, &dz, sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(f0, &ff0, sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dt, &ddt, sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(sslicex, &slicex, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(sslicey, &slicey, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(sslicez, &slicez, sizeof(int), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void**)&p0, DIMX * DIMY * DIMZ * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&vx, DIMX * DIMY * DIMZ * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&vy, DIMX * DIMY * DIMZ * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&vz, DIMX * DIMY * DIMZ * sizeof(float)));


	//float aa0 = 9 / 8;
	//float aa1 = 1 / 24;

	//load model
	float *temprho = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	float *tempalp = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	float *tempbuo = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	float *tempkap = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	float *tempzer = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));

	/*line = "modelrho.bin";
	char *inname4 = const_cast<char*>(line.c_str());
	FILE* file4 = fopen(inname4, "rb");
	for (int i = 0; i<(DIMX)*(DIMZ)*(DIMY); i++){
		float f4;
		fread(&f4, sizeof(float), 1, file4);
		temprho[i] = f4;
	}
	fclose(file4);

	line = "modelalp.bin";
	char *inname5 = const_cast<char*>(line.c_str());
	FILE* file4 = fopen(inname4, "rb");
	for (int i = 0; i<(DIMX)*(DIMZ)*(DIMY); i++){
		float f4;
		fread(&f4, sizeof(float), 1, file4);
		temprho[i] = f4;
	}
	fclose(file4);
	*/

	for (int kk = 0; kk < NZ; kk++)
	{
		for (int jj = 0; jj < NY; jj++)
		{
			for (int ii = 0; ii < NX - 1; ii++)
			{
				int ijk = ii + (jj*DIMY) + (kk*(DIMX*DIMY));
				temprho [ijk] = 1000;
				tempalp [ijk] = 1000;
				tempzer [ijk] = 0.0;
			}
		}
	}

	for (int kk = 0; kk < NZ; kk++)
	{
		for (int jj = 0; jj < NY; jj++)
		{
			for (int ii = 0; ii < NX; ii++)
			{
				int ijk = ii + (jj*DIMY) + (kk*(DIMX*DIMY));
				tempbuo [ijk] = 1/temprho[ijk];
				tempkap [ijk] = 1/tempalp[ijk];
			}
		}
	}

	FILE* file11 = fopen("mediumrho.ctxt", "wb");
	for (int kk = 0; kk < NZ; kk++)
	{
		for (int jj = 0; jj < NY; jj++)
		{
			for (int ii = 0; ii < NX; ii++)
			{
				int ijk = ii + (jj*DIMY) + (kk*(DIMX*DIMY));
				//outvz<<cvz[ij]<<" ";
				float f2 = temprho[ijk];
				fwrite(&f2, sizeof(float), 1, file11);
			}
		}
	}
	fclose(file11);

	dim3 blocks(DIMX/xthread,DIMY/ythread,DIMZ/zthread);
	dim3 threads(xthread, ythread, zthread);

	HANDLE_ERROR(cudaMalloc((void**)&ddx, sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&ddy, sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&ddz, sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&value_dp_dx, sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&value_dp_dy, sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&value_dp_dz, sizeof(float)));

	HANDLE_ERROR(cudaMalloc((void**)&kappaloc, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&buoloc, DIMX*DIMY*DIMZ*sizeof(float)));

	HANDLE_ERROR( cudaMemcpy( p0, tempzer,sizeof(float)*(DIMX*DIMY*DIMZ),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( vx, tempzer,sizeof(float)*(DIMX*DIMY*DIMZ),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( vy, tempzer,sizeof(float)*(DIMX*DIMY*DIMZ),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( vz, tempzer,sizeof(float)*(DIMX*DIMY*DIMZ),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( buoloc, tempbuo,sizeof(float)*(DIMX*DIMY*DIMZ),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( kappaloc, tempkap,sizeof(float)*(DIMX*DIMY*DIMZ),cudaMemcpyHostToDevice ) );

	free(temprho); free(tempkap); free(tempbuo); free(tempalp); free(tempzer);

	HANDLE_ERROR(cudaMalloc((void**)&sliceisox_vx, (DIMY*DIMZ)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&sliceisox_vz, (DIMY*DIMZ)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&sliceisoy_vx, (DIMX*DIMZ)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&sliceisoy_vz, (DIMX*DIMZ)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&sliceisoz_vx, (DIMX*DIMY)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&sliceisoz_vz, (DIMX*DIMY)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&seishorx_vx, DIMX*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&seishorx_vz, DIMX*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&seishory_vx, DIMY*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&seishory_vz, DIMY*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&seisvsp_vx, DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&seisvsp_vz, DIMZ*sizeof(float)));

	float *ssliceisox_vx = (float*)malloc(sizeof(float)*DIMY*DIMZ);
	float *ssliceisoy_vx = (float*)malloc(sizeof(float)*DIMX*DIMZ);
	float *ssliceisoz_vx = (float*)malloc(sizeof(float)*DIMX*DIMY);
	float *ssliceisox_vz = (float*)malloc(sizeof(float)*DIMY*DIMZ);
	float *ssliceisoy_vz = (float*)malloc(sizeof(float)*DIMX*DIMZ);
	float *ssliceisoz_vz = (float*)malloc(sizeof(float)*DIMX*DIMY);
	for (int jj=0; jj<DIMZ; jj++) {
		for (int ii=0; ii<DIMY; ii++) {
		int ij=DIMZ*jj + ii;
		ssliceisox_vx[ij] = 0.0;
		ssliceisox_vz[ij] = 0.0;
		}
		for (int ii=0; ii<DIMX; ii++) {
		int ij=DIMZ*jj + ii;
		ssliceisoy_vx[ij] = 0.0;
		ssliceisoy_vz[ij] = 0.0;
		}
	}
	for (int jj=0; jj<DIMY; jj++) {
		for (int ii=0; ii<DIMX; ii++) {
		int ij=DIMY*jj + ii;
		ssliceisoz_vx[ij] = 0.0;
		ssliceisoz_vz[ij] = 0.0;
		}
	}
	
	HANDLE_ERROR( cudaMemcpy( sliceisoy_vx, ssliceisoy_vx,sizeof(float)*(DIMX*DIMY*DIMZ),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( sliceisoy_vz, ssliceisoy_vz,sizeof(float)*(DIMX*DIMY*DIMZ),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( sliceisox_vx, ssliceisox_vx,sizeof(float)*(DIMX*DIMY*DIMZ),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( sliceisox_vz, ssliceisox_vz,sizeof(float)*(DIMX*DIMY*DIMZ),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( sliceisoz_vx, ssliceisoz_vx,sizeof(float)*(DIMX*DIMY*DIMZ),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( sliceisoz_vz, ssliceisoz_vz,sizeof(float)*(DIMX*DIMY*DIMZ),cudaMemcpyHostToDevice ) );

	int i_steps = 0;
	for (int nn=1; nn<(nite/nsamp); nn++)
	{
		for (int ns=1; ns<nsamp; ns++)
		{
			i_steps++;
			HANDLE_ERROR(cudaMemcpy(i_gpu, i_steps,sizeof(int),cudaMemcpyHostToDevice));
			kernel_p0<<<blocks,threads>>>(DDIMX, DDIMY, DDIMZ, vx, vy, vz, p0, ddx, ddy, ddz, kappaloc, buoloc, value_dvx_dx, value_dvy_dy, value_dvz_dz, is, js, ks, i_gpu, dt, a, f0);
			kernel_vx<<<blocks,threads>>>(DDIMX, DDIMY, DDIMZ, vx, p0, ddx, buoloc, value_dp_dx);
			kernel_vy<<<blocks,threads>>>(DDIMX, DDIMY, DDIMZ, vy, p0, ddy, buoloc, value_dp_dy);
			kernel_vz<<<blocks,threads>>>(DDIMX, DDIMY, DDIMZ, vz, p0, ddz, buoloc, value_dp_dz);
			kernel_select_trace << <blocks, threads >> >();
			kernel_save_seismo << <blocks, threads >> >();
		}
		kernel_save_snap<<<blocks,threads>>>(DDIMX, DDIMY, DDIMZ, sslicex, sslicey, sslicez, vx, vy, vz, sliceisox_vx, sliceisox_vz, sliceisoy_vx, sliceisoy_vz, sliceisoz_vx, sliceisoz_vz);
		HANDLE_ERROR( cudaMemcpy( ssliceisox_vx, sliceisox_vx,sizeof(float)*(DIMY*DIMZ),cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy( ssliceisox_vz, sliceisox_vz,sizeof(float)*(DIMY*DIMZ),cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy( ssliceisoy_vx, sliceisoy_vx,sizeof(float)*(DIMY*DIMZ),cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy( ssliceisoy_vz, sliceisoy_vz,sizeof(float)*(DIMY*DIMZ),cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy( ssliceisoz_vx, sliceisoz_vx,sizeof(float)*(DIMY*DIMZ),cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy( ssliceisoz_vz, sliceisoz_vz,sizeof(float)*(DIMY*DIMZ),cudaMemcpyDeviceToHost ) );
		
		char nmfile1[50];
		sprintf(nmfile1, "sliceisox_vx%05i.sxvxo",i_steps  );
		FILE* file1 = fopen (nmfile1, "wb");
		for (int jj=0; jj<DIMZ; jj++) {
			for (int ii=0; ii<DIMY; ii++) {
			int ij=DIMZ*jj + ii;
			float f1 = ssliceisox_vx[ij];
			fwrite(&f1, sizeof(float), 1, file1);
		}
		}
		fclose(file1);

		char nmfile2[50];
		sprintf(nmfile2, "sliceisox_vz%05i.sxvzo",i_steps  );
		FILE* file2 = fopen (nmfile2, "wb");
		for (int jj=0; jj<DIMZ; jj++) {
			for (int ii=0; ii<DIMY; ii++) {
			int ij=DIMZ*jj + ii;
			float f2 = ssliceisox_vz[ij];
			fwrite(&f2, sizeof(float), 1, file2);
		}
		}
		fclose(file2);

		char nmfile3[50];
		sprintf(nmfile3, "sliceisoy_vx%05i.syvxo",i_steps  );
		FILE* file3 = fopen (nmfile3, "wb");
		for (int jj=0; jj<DIMZ; jj++) {
			for (int ii=0; ii<DIMX; ii++) {
			int ij=DIMZ*jj + ii;
			float f3 = ssliceisoy_vx[ij];
			fwrite(&f3, sizeof(float), 1, file3);
		}
		}
		fclose(file3);

		char nmfile4[50];
		sprintf(nmfile4, "sliceisoy_vz%05i.syvzo",i_steps  );
		FILE* file4 = fopen (nmfile4, "wb");
		for (int jj=0; jj<DIMZ; jj++) {
			for (int ii=0; ii<DIMX; ii++) {
			int ij=DIMZ*jj + ii;
			float f4 = ssliceisoy_vz[ij];
			fwrite(&f4, sizeof(float), 1, file4);
		}
		}
		fclose(file4);

		char nmfile5[50];
		sprintf(nmfile5, "sliceisoz_vx%05i.szvxo",i_steps  );
		FILE* file5 = fopen (nmfile5, "wb");
		for (int jj=0; jj<DIMY; jj++) {
			for (int ii=0; ii<DIMX; ii++) {
			int ij=DIMY*jj + ii;
			float f5 = ssliceisoz_vx[ij];
			fwrite(&f5, sizeof(float), 1, file5);
		}
		}
		fclose(file3);

		char nmfile6[50];
		sprintf(nmfile6, "sliceisoz_vz%05i.szvzo",i_steps  );
		FILE* file6 = fopen (nmfile6, "wb");
		for (int jj=0; jj<DIMY; jj++) {
			for (int ii=0; ii<DIMX; ii++) {
			int ij=DIMY*jj + ii;
			float f6 = ssliceisoz_vz[ij];
			fwrite(&f6, sizeof(float), 1, file6);
		}
		}
		fclose(file6);
	}

	HANDLE_ERROR(cudaFree(p0));
	HANDLE_ERROR(cudaFree(vx));
	HANDLE_ERROR(cudaFree(vy));
	HANDLE_ERROR(cudaFree(vz));
	HANDLE_ERROR(cudaFree(kappaloc));
	HANDLE_ERROR(cudaFree(buoloc));

	printf ("Selesai...");
	getch();
}
