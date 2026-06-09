#ifndef GRID2D_H
#define GRID2D_H
void update_grid(double **u,double **u_new, int rows, int cols);
void update_grid_hallo(double **u,double **u_new, int rows, int cols);
double calc_diff(double **u,double **u_new, int rows, int cols);

#endif
