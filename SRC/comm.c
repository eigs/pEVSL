#include "pevsl_protos.h"

/* Constructor */
int pEVSL_CommCreate(pevsl_Comm *comm, MPI_Comm comm_global, int ngroups) {

    int err;
    double t_s, t_e;

    t_s = MPI_Wtime();

    MPI_Comm_size(comm_global, &comm->global_size);
    MPI_Comm_rank(comm_global, &comm->global_rank);

    if (ngroups > comm->global_size) {
        printf("Warning: Number of procs is %d. Number of groups asked was %d, now changed to %d\n",\
                comm->global_size, ngroups, comm->global_size);
        ngroups = comm->global_size;
    }

    comm->comm_global = comm_global;
    comm->ngroups = ngroups;

    // Create communicators for ngroups groups
    pEVSL_Part1d(comm->global_size, ngroups, &comm->group_id, &comm->global_rank, &comm->group_size, 2);
    err = MPI_Comm_split(comm_global, comm->group_id, 1, &comm->comm_group); PEVSL_CHKERR(err != MPI_SUCCESS);
    MPI_Comm_rank(comm->comm_group, &comm->group_rank);

    // Create a communicator for the ``group leaders'' that are the procs of rank 0 in all groups
    int color;
    if (comm->group_rank) {
        color = MPI_UNDEFINED;
    } else {
        color = 0;
    }

    err = MPI_Comm_split(comm_global, color, comm->group_id, &comm->comm_group_leader); PEVSL_CHKERR(err != MPI_SUCCESS);

    if (comm->group_rank) {
        PEVSL_CHKERR(comm->comm_group_leader != MPI_COMM_NULL);
    } else {
        int leader_rank, leader_size;
        MPI_Comm_size(comm->comm_group_leader, &leader_size);
        MPI_Comm_rank(comm->comm_group_leader, &leader_rank);
        PEVSL_CHKERR(leader_size != comm->ngroups);
        PEVSL_CHKERR(leader_rank != comm->group_id);
    }

    t_e = MPI_Wtime();
    pevsl_stat.t_comm_create = t_e - t_s;

    return 0;
}


/* Destructor */
void pEVSL_CommFree(pevsl_Comm *comm) {
    int err;
    err = MPI_Comm_free(&comm->comm_group);
    PEVSL_CHKERR(err != MPI_SUCCESS);

    if (comm->comm_group_leader != MPI_COMM_NULL) {
        err = MPI_Comm_free(&comm->comm_group_leader);
        PEVSL_CHKERR(err != MPI_SUCCESS);
    }
}

