#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>



/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

void copyfunc(struct particles* part, struct particles* particlesGPU)
{
    FPpart* dev_x, * dev_y, * dev_z, * dev_u, * dev_v, * dev_w, * dev_q;
    size_t size_device = part->npmax * sizeof(FPpart);

    cudaMalloc(&dev_x, size_device);
    cudaMalloc(&dev_y, size_device);
    cudaMalloc(&dev_z, size_device);
    cudaMalloc(&dev_u, size_device);
    cudaMalloc(&dev_v, size_device);
    cudaMalloc(&dev_w, size_device);
    cudaMalloc(&dev_q, size_device);

    cudaMemcpy(dev_x, part->x, size_device, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, part->y, size_device, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_z, part->z, size_device, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_u, part->u, size_device, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v, part->v, size_device, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_w, part->w, size_device, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_q, part->q, size_device, cudaMemcpyHostToDevice);

    cudaMemcpy(&(particlesGPU->x), &dev_x, sizeof(particlesGPU->x), cudaMemcpyHostToDevice);
    cudaMemcpy(&(particlesGPU->y), &dev_y, sizeof(particlesGPU->y), cudaMemcpyHostToDevice);
    cudaMemcpy(&(particlesGPU->z), &dev_z, sizeof(particlesGPU->z), cudaMemcpyHostToDevice);
    cudaMemcpy(&(particlesGPU->u), &dev_u, sizeof(particlesGPU->u), cudaMemcpyHostToDevice);
    cudaMemcpy(&(particlesGPU->v), &dev_v, sizeof(particlesGPU->v), cudaMemcpyHostToDevice);
    cudaMemcpy(&(particlesGPU->w), &dev_w, sizeof(particlesGPU->w), cudaMemcpyHostToDevice);
    cudaMemcpy(&(particlesGPU->q), &dev_q, sizeof(particlesGPU->q), cudaMemcpyHostToDevice);
}

__device__ void subcycling(particles * part, EMfield * field, grid * grd, parameters * param, int idxX)
{
            // auxiliary variables
            FPpart dt_sub_cycling = (FPpart)param->dt / ((double)part->n_sub_cycles);//r
            FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;//r
            FPpart omdtsq, denom, ut, vt, wt, udotb;//r

            // local (to the particle) electric and magnetic field
            FPfield Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;//r

            // interpolation densities
            int ix, iy, iz;//r
            FPfield weight[2][2][2];//r
            FPfield xi[2], eta[2], zeta[2];//r

            // intermediate particle position and velocity
            FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;


            xptilde = part->x[idxX];
            yptilde = part->y[idxX];
            zptilde = part->z[idxX];


            // calculate the average velocity iteratively
            for (int innter = 0; innter < part->NiterMover; innter++) {
                // interpolation G-->P
                ix = 2 + int((part->x[idxX] - grd->xStart) * grd->invdx);
                iy = 2 + int((part->y[idxX] - grd->yStart) * grd->invdy);
                iz = 2 + int((part->z[idxX] - grd->zStart) * grd->invdz);

                // calculate weights
                long xi0_index_flat = get_idx(ix - 1, iy, iz, grd->nyn, grd->nzn);// from Alloc.h
                xi[0] = part->x[idxX] - grd->XN_flat[xi0_index_flat];// from Grid.h
                long eta0_index_flat = get_idx(ix, iy - 1, iz, grd->nyn, grd->nzn);// from Alloc.h
                eta[0] = part->y[idxX] - grd->YN_flat[eta0_index_flat];// from Grid.h
                long zeta0_index_flat = get_idx(ix, iy, iz - 1, grd->nyn, grd->nzn);// from Alloc.h
                zeta[0] = part->z[idxX] - grd->ZN_flat[zeta0_index_flat];// from Grid.h
                long index_flat_1 = get_idx(ix, iy, iz, grd->nyn, grd->nzn);// from Alloc.h
                xi[1] = grd->XN_flat[index_flat_1] - part->x[idxX];// from Grid.h
                eta[1] = grd->YN_flat[index_flat_1] - part->y[idxX];// from Grid.h
                zeta[1] = grd->ZN_flat[index_flat_1] - part->z[idxX];// from Grid.h
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

                // set to zero local electric and magnetic field
                Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++) {
                            long index_flat = get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn);
                            Exl += weight[ii][jj][kk] * field->Ex_flat[index_flat];
                            Eyl += weight[ii][jj][kk] * field->Ey_flat[index_flat];
                            Ezl += weight[ii][jj][kk] * field->Ez_flat[index_flat];
                            Bxl += weight[ii][jj][kk] * field->Bxn_flat[index_flat];
                            Byl += weight[ii][jj][kk] * field->Byn_flat[index_flat];
                            Bzl += weight[ii][jj][kk] * field->Bzn_flat[index_flat];
                        }

                // end interpolation
                omdtsq = qomdt2 * qomdt2 * (Bxl * Bxl + Byl * Byl + Bzl * Bzl);
                denom = 1.0 / (1.0 + omdtsq);
                // solve the position equation
                ut = part->u[idxX] + qomdt2 * Exl;
                vt = part->v[idxX] + qomdt2 * Eyl;
                wt = part->w[idxX] + qomdt2 * Ezl;
                udotb = ut * Bxl + vt * Byl + wt * Bzl;
                // solve the velocity equation
                uptilde = (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
                vptilde = (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
                wptilde = (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;
                // update position
                part->x[idxX] = xptilde + uptilde * dto2;
                part->y[idxX] = yptilde + vptilde * dto2;
                part->z[idxX] = zptilde + wptilde * dto2;


            } // end of iteration
            // update the final position and velocity
            part->u[idxX] = 2.0 * uptilde - part->u[idxX];
            part->v[idxX] = 2.0 * vptilde - part->v[idxX];
            part->w[idxX] = 2.0 * wptilde - part->w[idxX];
            part->x[idxX] = xptilde + uptilde * dt_sub_cycling;
            part->y[idxX] = yptilde + vptilde * dt_sub_cycling;
            part->z[idxX] = zptilde + wptilde * dt_sub_cycling;


            //////////
            //////////
            ////////// BC

            // X-DIRECTION: BC particles
            if (part->x[idxX] > grd->Lx) {
                if (param->PERIODICX == true) { // PERIODIC
                    part->x[idxX] = part->x[idxX] - grd->Lx;
                }
                else { // REFLECTING BC
                    part->u[idxX] = -part->u[idxX];
                    part->x[idxX] = 2 * grd->Lx - part->x[idxX];
                }
            }

            if (part->x[idxX] < 0) {
                if (param->PERIODICX == true) { // PERIODIC
                    part->x[idxX] = part->x[idxX] + grd->Lx;
                }
                else { // REFLECTING BC
                    part->u[idxX] = -part->u[idxX];
                    part->x[idxX] = -part->x[idxX];
                }
            }

            // Y-DIRECTION: BC particles
            if (part->y[idxX] > grd->Ly) {
                if (param->PERIODICY == true) { // PERIODIC
                    part->y[idxX] = part->y[idxX] - grd->Ly;
                }
                else { // REFLECTING BC
                    part->v[idxX] = -part->v[idxX];
                    part->y[idxX] = 2 * grd->Ly - part->y[idxX];
                }
            }

            if (part->y[idxX] < 0) {
                if (param->PERIODICY == true) { // PERIODIC
                    part->y[idxX] = part->y[idxX] + grd->Ly;
                }
                else { // REFLECTING BC
                    part->v[idxX] = -part->v[idxX];
                    part->y[idxX] = -part->y[idxX];
                }
            }

            // Z-DIRECTION: BC particles
            if (part->z[idxX] > grd->Lz) {
                if (param->PERIODICZ == true) { // PERIODIC
                    part->z[idxX] = part->z[idxX] - grd->Lz;
                }
                else { // REFLECTING BC
                    part->w[idxX] = -part->w[idxX];
                    part->z[idxX] = 2 * grd->Lz - part->z[idxX];
                }
            }

            if (part->z[idxX] < 0) {
                if (param->PERIODICZ == true) { // PERIODIC
                    part->z[idxX] = part->z[idxX] + grd->Lz;
                }
                else { // REFLECTING BC
                    part->w[idxX] = -part->w[idxX];
                    part->z[idxX] = -part->z[idxX];
                }
            }
}

// GPU version 
__global__ void mover_PC_gpu(particles* parts, EMfield* field, grid* grd, parameters* param)
{
    int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    int idxY = blockIdx.y * blockDim.y + threadIdx.y;

    particles* part = &(parts[idxY]);

    if (idxX < part->nop)

    {
        subcycling(part,field,grd,param,idxX);
    }
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        for (int = 0; i < part -> nop; i++){
            
        subcycling (part, field,grd,param,i);
        }                                                            
    return(0); // exit succcesfully
} // 



/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
