/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"

#define Db 512



int main(int argc, char **argv){
    
    // Read the inputfile and fill the param structure
    parameters param;
    parameters bb_param;
    // Read the input file name from command line
    readInputFile(&param,argc,argv);
    printParameters(&param);
    saveParameters(&param);
    
    // Timing variables
    double iStart = cpuSecond();
    double iMover, iInterp, eMover = 0.0, eInterp= 0.0;
    
    // Set-up the grid information
    grid grd;
    grid bb_grd;
    setGrid(&param, &grd);
    
    // Allocate Fields
    EMfield field;
    EMfield bb_field;
    field_allocate(&grd,&field);
    EMfield_aux field_aux;
    field_aux_allocate(&grd,&field_aux);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids = new interpDensSpecies[param.ns];
    for (int is=0; is < param.ns; is++)
        interp_dens_species_allocate(&grd,&ids[is],is);
    // Net densities
    interpDensNet idn;
    interp_dens_net_allocate(&grd,&idn);
    
    // Allocate Particles
    particles *part = new particles[param.ns];
    particles *bb_part = new particles[bb_param.ns];
    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param,&part[is],is);
    }
    
    // Initialization
    initGEM(&param,&grd,&field,&field_aux,part,ids);
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    
    particles *particlesGPU = new particles[param.ns];
    cudaMalloc(&particlesGPU, sizeof(particles) * param.ns);
    cudaMemcpy(particlesGPU, part, sizeof(particles) * param.ns, cudaMemcpyHostToDevice);
    for (int is=0; is < param.ns; is++) copyfunc(&part[is], &particlesGPU[is]); 
    
    EMfield *fieldGPU;
    cudaMalloc(&fieldGPU, sizeof(EMfield));
    cudaMemcpy(fieldGPU, &field, sizeof(EMfield), cudaMemcpyHostToDevice);
    
     // field
    FPfield *dev_fieldEx, *dev_fieldEy, *dev_fieldEz, *dev_fieldBxn, *dev_fieldByn, *dev_fieldBzn;

    cudaMalloc(&dev_fieldEx, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dev_fieldEy, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dev_fieldEz, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dev_fieldBxn, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dev_fieldByn, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dev_fieldBzn, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));

    cudaMemcpy(dev_fieldEx, field->Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_fieldEy, field->Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_fieldEz, field->Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_fieldBxn, field->Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_fieldByn, field->Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_fieldBzn, field->Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMemcpy(&(fieldGPU->Ex_flat), &dev_fieldEx, sizeof(fieldGPU->Ex_flat), cudaMemcpyHostToDevice);
    cudaMemcpy(&(fieldGPU->Ey_flat), &dev_fieldEy, sizeof(fieldGPU->Ey_flat), cudaMemcpyHostToDevice);
    cudaMemcpy(&(fieldGPU->Ez_flat), &dev_fieldEz, sizeof(fieldGPU->Ez_flat), cudaMemcpyHostToDevice);
    cudaMemcpy(&(fieldGPU->Bxn_flat), &dev_fieldBxn, sizeof(fieldGPU->Bxn_flat), cudaMemcpyHostToDevice);
    cudaMemcpy(&(fieldGPU->Byn_flat), &dev_fieldByn, sizeof(fieldGPU->Byn_flat), cudaMemcpyHostToDevice);
    cudaMemcpy(&(fieldGPU->Bzn_flat), &dev_fieldBzn, sizeof(fieldGPU->Bzn_flat), cudaMemcpyHostToDevice);
    
    // field end 
    
    struct grid *grdGPU;
    cudaMalloc(&grdGPU, sizeof(grid));
    cudaMemcpy(grdGPU, &grd, sizeof(grid), cudaMemcpyHostToDevice);
      // grid
    
    FPfield *dev_grdXN, *dev_grdYN, *dev_grdZN;

    cudaMalloc(&dev_grdXN, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dev_grdYN, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dev_grdZN, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));

    cudaMemcpy(dev_grdXN, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_grdYN, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_grdZN, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMemcpy(&(grdGPU->XN_flat), &dev_grdXN, sizeof(grdGPU->XN_flat), cudaMemcpyHostToDevice);
    cudaMemcpy(&(grdGPU->YN_flat), &dev_grdYN, sizeof(grdGPU->YN_flat), cudaMemcpyHostToDevice);
    cudaMemcpy(&(grdGPU->ZN_flat), &dev_grdZN, sizeof(grdGPU->ZN_flat), cudaMemcpyHostToDevice);
    
    // grid end 
    
    parameters *paramGPU;
    cudaMalloc(&paramGPU, sizeof(parameters));
    cudaMemcpy(paramGPU, &param, sizeof(parameters), cudaMemcpyHostToDevice);
    
     //EMfield *fieldGPU;
    //cudaMalloc(&fieldGPU, sizeof(EMfield));
    //cudaMemcpy(fieldGPU, &field, sizeof(EMfield), cudaMemcpyHostToDevice);
    //field_copy_cpu2gpu(&grd, &field, fieldGPU);  // Correct the pointers of the arrays

    //struct grid *grdGPU;
    //cudaMalloc(&grdGPU, sizeof(grid));
    //cudaMemcpy(grdGPU, &grd, sizeof(grid), cudaMemcpyHostToDevice);
    //setGridGPU(&param, &grd, grdGPU);  // Correct the pointers of the arrays

    //parameters *paramGPU;
    //cudaMalloc(&paramGPU, sizeof(parameters));
    //cudaMemcpy(paramGPU, &param, sizeof(parameters), cudaMemcpyHostToDevice);
    
    
    // not neccesary 
    //interpDensSpecies *idsGPU = new interpDensSpecies[param.ns]; // this dont in use
    //interpDensSpecies *idsGPU2CPU = new interpDensSpecies[param.ns];
    //cudaMalloc(&idsGPU, sizeof(interpDensSpecies) * param.ns);
    //for (int is=0; is < param.ns; is++)
        //interp_dens_species_copy_cpu2gpu(&grd, &ids[is], &idsGPU[is]);  // Correct the pointers of the arrays
    //std::memcpy(idsGPU2CPU, &ids, sizeof(interpDensSpecies) * param.ns);  // cudaMemcpy is done later (in every iteration)
    // not neccesary 
    
    int largestNumParticles = 0;
    for (int i = 0; i < param.ns; i++) {
        if (part[i].nop > largestNumParticles) {
            largestNumParticles = part[i].nop;
        }
    }

    
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn,ids,&grd,param.ns);
        
        
        
        // implicit mover
        iMover = cpuSecond(); // start timer for mover
        //for (int is=0; is < param.ns; is++)
            //mover_PC(&part[is],&field,&grd,&param);  // kernel launch 
        eMover += (cpuSecond() - iMover); // stop timer for mover
        
        // wrapper goes here 
        //////////////////////////////////////////////////////////////
        gpu_mover_PC<<<dim3(largestNumParticles / Db + 1, 1, 1), dim3(TpBx, param->ns, 1)>>>(particlesGPU, fieldGPU, grdGPU, paramGPU);
        
        //gpu_mover_PC_wrapper(particlesGPU, fieldGPU, grdGPU, paramGPU, largestNumParticles);
        ///////////////////////////////////////////////////////////////
        // bring back part, field, grd and param 
        // bb =  vrought back to host (from device)
        cudaMemcpy (bb_part,particlesGPU, sizeof(particles), cudaMemcpyDeviceToHost);
        cudaMemcpy (bb_field,fieldGPU, sizeof(EMfield), cudaMemcpyDeviceToHost);
        cudaMemcpy (bb_grd,grdGPU, sizeof(grd), cudaMemcpyDeviceToHost);
        cudaMemcpy (bb_param,paramGPU, sizeof(parameters), cudaMemcpyDeviceToHost);
            
        
        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is=0; is < param.ns; is++)
            interpP2G(&bb_part[is],&ids[is],&grd); // for bb
        
        
        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++)
            applyBCids(&ids[is],&bb_grd,&bb_param);
        // sum over species
        sumOverSpecies(&idn,ids,&bb_grd,bb_param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon,&grd,&bb_param);
        
        
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &bb_grd,&bb_field);
            VTK_Write_Scalars(cycle, &bb_grd,ids,&idn);
        }
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation
        
        
    
    }  // end of one PIC cycle
    
    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    // interp
    interp_dens_net_deallocate(&grd,&idn);
    
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        particle_deallocate(&part[is]);
    }
    
    cudaFree (particlesGPU);
    cudaFree (fieldGPU);
    cudaFree (grdGPU);                
    cudaFree (paramGPU);         
                    
    // stop timer
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;
    
    // exit
    return 0;
}


