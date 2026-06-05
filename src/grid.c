#include <math.h>
#include "grid.h"

void update_grid(double **u, double **u_new, int rows, int cols)
{
    for(int i=2;i<rows-2;i++){
        for(int j=2;j<cols-2;j++){
            u_new[i][j]=0.25*(u[i+1][j]+u[i-1][j] + u[i][j+1]+u[i][j-1]);
        }
    }
}

void update_grid_hallo(double **u, double **u_new, int rows, int cols)
{
    int i=1;
    for(int j=1;j<cols-1;j++) {
        u_new[i][j]=0.25*(u[i+1][j]+u[i-1][j]+u[i][j+1]+u[i][j-1]);
    }

    i = rows-2;
    for (int j=1;j<cols-1;j++) {
        u_new[i][j]=0.25*(u[i+1][j]+u[i-1][j]+u[i][j+1]+u[i][j-1]);
    }

    int j=1;
    for(int ii=1;ii<rows-1;ii++) {
        u_new[ii][j]=0.25*(u[ii+1][j]+u[ii-1][j]+u[ii][j+1] + u[ii][j-1]);
    }

    j=cols-2;
    for (int ii=1;ii<rows-1;ii++){
        u_new[ii][j]=0.25*(u[ii+1][j]+u[ii-1][j]+u[ii][j+1] + u[ii][j-1]);
    }
}

double calc_diff(double **u, double **u_new, int rows, int cols)
{
    double max_diff=0;

    for(int i=1;i<rows-1;i++) {
        for(int j=1;j<cols-1;j++) {
            double diff=fabs(u_new[i][j]-u[i][j]);
            if (diff>max_diff){
                max_diff=diff;
            }
        }
    }
    return max_diff;
}

