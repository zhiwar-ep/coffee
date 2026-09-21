import os
from yade import mpy as mp
from yade import export 
import math
from collections import defaultdict
import numpy as np
from mpi4py import MPI
from yade import pack
comm = mp.comm_slave

#print("YADE: current working directory:", os.getcwd())

writeInterval = 0.5   # seconds ( = OpenFOAM writeInterval controlDict )
endTime = 900.0


spring_frame_number = 0
sphere_frame_number = 0


yadeDtFixed = 5e-6 #5e-6
O.dynDt = False
O.dt    = yadeDtFixed


timeRatio = int(round(writeInterval / yadeDtFixed))
timeRatio = max(1, timeRatio)


if mp.rank == 0:
    print("[EXPORT TIMEs] yadeDtFixed=", yadeDtFixed, " writeInterval=", writeInterval, " iterPeriod=", timeRatio)



parallelYade=True 
numProcOF=4



O.periodic = False # if true, define O.cell.setBox
#O.cell.setBox(0.01000005, 0.0100005, 0.0100005)   #comment out if O.periodic = False
numspheres = 400
young = 25e6 #5e6
density = 1200
NSTEPS = int(round(endTime / yadeDtFixed))
#NSTEPS = 12000000 #int(2e5)
fixed_seed = 2 # to keep random arrangment of spheres repertable. if changing this value, the arrangment will be different.



O.materials.append(FrictMat(young=young, poisson=0.5, frictionAngle=radians(15), density=density, label='spheremat'))
O.materials.append(FrictMat(young=young * 100, poisson=0.5, frictionAngle=0, density=0, label='wallmat'))



# size of the domain  containing the spheres
maxvalX = 0.02
maxvalY = 0.01
maxvalZ = 0.01

hx, hy, hz = maxvalX/2., maxvalY/2., maxvalZ/2.


# -----------------------
# Coffee bean geometry
# -----------------------

L = 0.009
W = 0.006
T = 0.004


rRelFuzz = 0.0

particle_radius = 1e-5


def insideCoffeeBean(pt):
    x, y, z = pt

    X = x / (L * 0.5)
    Y = y / (T * 0.5)
    Z = z / (W * 0.5)

    base = X**2 + Y**2 + Z**2

    grooveDepth = 0.6
    grooveWidth = 0.5

    Y_eff = Y + grooveDepth * math.exp(
        -(Z / grooveWidth)**2
    )

    bean = X**2 + Y_eff**2 + Z**2

    return (base <= 1.0) and (bean <= 1.0)



particle_radius = 1e-5      # very small radius spheres
placement_radius = 0.0002 # for placement of spheres assume larger spheres to prevent coagulation of spheres in one place
reference_radius = 0.00075  # radius of a more stable spheres
effective_mass = (density * (4.0 / 3.0) * math.pi * reference_radius**3)

effective_inertia = ((2.0 / 5.0) * effective_mass * reference_radius**2)


sp = pack.SpherePack()

sp.makeCloud(
    (-L/2, -T/2, -W/2),
    ( L/2,  T/2,  W/2),
    rMean=placement_radius,
    rRelFuzz=0,
    num=numspheres,
    periodic=False,
    seed=fixed_seed
)




spBean = pack.SpherePack()

for center, rad in sp:
    if insideCoffeeBean(center):
        spBean.add(center, particle_radius)

print("Coffee bean spheres:", len(spBean))

O.bodies.append([
    sphere(center, rad, material='spheremat')
    for center, rad in spBean
])

for b in O.bodies:
    if isinstance(b.shape, Sphere):
        b.state.mass = effective_mass
        b.state.inertia = Vector3(
            effective_inertia,
            effective_inertia,
            effective_inertia
        )




# spring creation 
#----------------
spring_damping = 1.0
mass = 0.000268                          # same as wood sample
spring_k = (spring_damping**2) / (4 * mass)
min_elongation_percent = 80             # freeze threshold (compression)
max_elongation_percent = 1000               # break threshold (expansion)

print(f"[SPRINGS] spring_k = {spring_k:.6e}")

n_neighbors = 6
springsList = {}                 # springID : (i,j,L0)
broken_springs = []
existing_pairs = set()
spring_id_counter = 0


# collect sphere IDs
sphereIDs = [b.id for b in O.bodies if isinstance(b.shape, Sphere)]
positions  = [O.bodies[i].state.pos for i in sphereIDs]

# KD-tree for nearest neighboring spheres
from scipy.spatial import cKDTree
tree = cKDTree(positions)

for idx, sid in enumerate(sphereIDs):
    dists, idxs = tree.query(positions[idx], k=n_neighbors+1)  # +1 for itself

    for k in range(1, len(idxs)):      # skip itself
        sid2 = sphereIDs[idxs[k]]
        pair = tuple(sorted((sid, sid2)))
        if pair in existing_pairs:
            continue

        existing_pairs.add(pair)
        L0 = dists[k]

        springsList[spring_id_counter] = (pair[0], pair[1], L0)
        spring_id_counter += 1

print(f"[SPRINGS] Created {len(springsList)} springs.")

degree = {sid: 0 for sid in sphereIDs}

for spring_id, (i, j, L0) in springsList.items():
    degree[i] += 1
    degree[j] += 1

print("\n[SPRING DEGREE CHECK]")
for sid in sorted(degree):
    print(f"sphere {sid}: {degree[sid]} springs")

print(
    "[SPRING DEGREE] min =",
    min(degree.values()),
    "max =",
    max(degree.values())
)

# save initial springs list 

os.makedirs("springs", exist_ok=True)
with open("springs/springsList.txt", "w") as f:
    f.write("springID sphere1 sphere2 length\n")
    for sid, (i,j,L0) in springsList.items():
        f.write(f"{sid} {i} {j} {L0:.6e}\n")



# send springsList to all MPI RANKS ( broadcast) 
# ----------------------------------------

if mp.rank == 0:
    springs_serial = [(sid, data[0], data[1], data[2]) for sid, data in springsList.items()]
else:
    springs_serial = None

# broadcast to all ranks
springs_serial = mp.comm.bcast(springs_serial, root=0)

# reconstruct springsList on ALL ranks
springsList = { sid : (i, j, L0) for (sid, i, j, L0) in springs_serial }

if mp.rank != 0:
    print(f"[rank {mp.rank}] received {len(springsList)} springs")



springsLambda = { sid: 1.0 for sid in springsList }  # per-spring lambda storage

# broadcast springsLambda to all MPI ranks
if mp.rank == 0:
    springsLambda_serial = [(sid, springsLambda[sid]) for sid in springsLambda]
else:
    springsLambda_serial = None

springsLambda_serial = mp.comm.bcast(springsLambda_serial, root=0)

springsLambda = { sid: lam for (sid, lam) in springsLambda_serial }

if mp.rank != 0:
    print(f"[rank {mp.rank}] received springsLambda for {len(springsLambda)} springs")



def apply_spring_forces():
    """
    Gather sphere positions/lambdaDot from all MPI ranks.
    Rank 0 computes ALL spring forces using the global particle data.
    Then forces are broadcast back to all ranks.
    """

    global springsList, broken_springs, springsLambda

    if O.iter == 2:
        print("[CHECK] apply_spring_forces is running on rank", mp.rank)

    # ---------------------------------------------------------
    # 1. Each rank sends all spheres that it can currently see
    # ---------------------------------------------------------
    local_particles = []

    for b in O.bodies:
        if isinstance(b.shape, Sphere):

            p = b.state.pos

            local_particles.append(
                (
                    b.id,
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    float(b.state.lambdaDot)
                )
            )

    # Gather all sphere data on rank 0
    all_particles = mp.comm.gather(local_particles, root=0)


    # ---------------------------------------------------------
    # 2. Rank 0 reconstructs global particle information
    #    and calculates ALL springs
    # ---------------------------------------------------------
    if mp.rank == 0:

        particle_data = {}

        for rank_particles in all_particles:
            for sid, x, y, z, lambdaDot in rank_particles:

                # A sphere can appear on more than one rank because
                # of MPI ghost copies. Keep only one copy.
                if sid not in particle_data:
                    particle_data[sid] = (
                        Vector3(x, y, z),
                        lambdaDot
                    )


        force_map = defaultdict(lambda: Vector3(0, 0, 0))
        to_remove = []

        missing_springs = 0


        for spring_id, (i, j, L0) in springsList.items():

            # Both particles should now be available globally
            if i not in particle_data or j not in particle_data:
                missing_springs += 1
                continue


            pi, lambdaDot_i = particle_data[i]
            pj, lambdaDot_j = particle_data[j]


            # -------------------------------------------------
            # Geometry
            # -------------------------------------------------
            delta = pj - pi
            L_current = delta.norm()

            if L_current == 0:
                continue

            direction = delta.normalized()


            # -------------------------------------------------
            # FREEZE under strong compression
            # -------------------------------------------------
            min_L = (min_elongation_percent / 100.0) * L0

            if L_current <= min_L:
                continue


            # -------------------------------------------------
            # BREAK under excessive extension
            # -------------------------------------------------
            elong_pct = ((L_current - L0) / L0) * 100.0

            if elong_pct > max_elongation_percent:

                broken_springs.append(
                    (
                        spring_id,
                        i,
                        j,
                        O.time,
                        L0,
                        L_current,
                        L_current - L0,
                        elong_pct
                    )
                )

                to_remove.append(spring_id)
                continue


            # -------------------------------------------------
            # Dynamic lambda
            # -------------------------------------------------
            lambdaDot_avg = (0.5 * (lambdaDot_i + lambdaDot_j))
            lambdaDot_avg *= 1


            lambda_prev = springsLambda[spring_id]

            lambda_new = (
                lambda_prev
                + O.dt * lambdaDot_avg
            )


            # clamp
            lambda_new = max(
                0.1,
                min(lambda_new, 2.0)
            )

            springsLambda[spring_id] = lambda_new


            # -------------------------------------------------
            # Target spring length
            # -------------------------------------------------
            target_L = L0 * lambda_new


            # -------------------------------------------------
            # Hooke force
            # -------------------------------------------------
            lambda_eff = L_current - target_L

            Fvec = (
                spring_k
                * lambda_eff
                * direction
            )


            # Equal and opposite forces
            force_map[i] += Fvec
            force_map[j] -= Fvec


        # convert Vector3 to ordinary tuples for MPI
        force_data = [
            (
                sid,
                float(F[0]),
                float(F[1]),
                float(F[2])
            )
            for sid, F in force_map.items()
        ]


        if missing_springs > 0:
            print(
                "[SPRING WARNING]",
                missing_springs,
                "springs have missing global particle data"
            )

    else:

        force_data = None
        to_remove = None


    # ---------------------------------------------------------
    # 3. Broadcast results to every MPI rank
    # ---------------------------------------------------------
    force_data = mp.comm.bcast(force_data, root=0)
    to_remove = mp.comm.bcast(to_remove, root=0)

    # Keep lambda values identical on all MPI ranks
    springsLambda = mp.comm.bcast(
        springsLambda if mp.rank == 0 else None,
        root=0
    )


    # ---------------------------------------------------------
    # 4. Apply calculated forces wherever that sphere exists
    # ---------------------------------------------------------
    for sid, fx, fy, fz in force_data:

        if sid < len(O.bodies):

            b = O.bodies[sid]

            if b is not None and isinstance(b.shape, Sphere):

                O.forces.addF(
                    sid,
                    Vector3(fx, fy, fz)
                )


    
    # 5. Remove broken springs consistently from ALL ranks
    # ---------------------------------------------------------
    for spring_id in to_remove:

        if spring_id in springsList:
            del springsList[spring_id]

        if spring_id in springsLambda:
            del springsLambda[spring_id]

#----------


fluidCoupling = FoamCoupling()
fluidCoupling.couplingModeParallel = parallelYade
fluidCoupling.isGaussianInterp = True


sphereIDs = [b.id for b in O.bodies if type(b.shape) == Sphere]


fluidCoupling.SetOpenFoamSolver("porousGasificationFoam", numProcOF)




# tell the coupling engine which IDs to couple
fluidCoupling.setIdList(sphereIDs)
fluidCoupling.setNumParticles(len(sphereIDs))





def printAndSaveDtInfo():
    yadeDt = O.dt
    foamDt = fluidCoupling.foamDeltaT
    ratio  = foamDt / yadeDt if yadeDt > 0 else float("inf")

    msg = (f"[Time Step INFO] iter={O.iter}  time={O.time:.6e}  "
           f"yadeDt={yadeDt:.6e}  foamDt={foamDt:.6e}  ratio={ratio:.6f}")

    # print once in terminal (rank 0 only)
    if mp.rank == 0:
        #print(msg)

        # save to file
        with open("dtInfo.txt", "w") as f:
            f.write("iter time yadeDt foamDt ratio\n")
            f.write(f"{O.iter} {O.time:.6e} {yadeDt:.6e} {foamDt:.6e} {ratio:.6f}\n")




def printStep():
	print("step = ", O.iter)


def savePos():
    export.text(f"spheres/spheres_{O.iter:05d}.txt")
    
export.text("spheres/spheres_0.txt")
#export.VTKExporter("spheres/vtk-0").exportSpheres()


# to list all vtk files in a single pvd file with time stamps. in paraview only open this pvd file not all vtk files.

pvd_spheres = []   # list of (time, filename)
pvd_springs = []

def write_pvd(pvd_path, entries):
    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        for t, fname in entries:
            f.write(f'    <DataSet timestep="{t:.6f}" file="{fname}"/>\n')
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')


# Export springs as VTK ( not VTKRecorder )
# ---------------------
def export_springs():

    global spring_frame_number
    global pvd_springs

    os.makedirs("springs", exist_ok=True)

    # ---------------------------------------------------------
    # 1. Each MPI rank sends all sphere positions it can see
    # ---------------------------------------------------------
    local_particles = []

    for b in O.bodies:

        if isinstance(b.shape, Sphere):

            p = b.state.pos

            local_particles.append(
                (
                    b.id,
                    float(p[0]),
                    float(p[1]),
                    float(p[2])
                )
            )


    # Gather particle positions on rank 0
    all_particles = mp.comm.gather(
        local_particles,
        root=0
    )


    # ---------------------------------------------------------
    # 2. Rank 0 reconstructs GLOBAL sphere positions
    # ---------------------------------------------------------
    if mp.rank == 0:

        global_positions = {}

        for rank_particles in all_particles:

            for sid, x, y, z in rank_particles:

                # Same particle may exist as a ghost on several ranks.
                # Keep only one copy.
                if sid not in global_positions:

                    global_positions[sid] = Vector3(
                        x,
                        y,
                        z
                    )


        # -----------------------------------------------------
        # 3. Build ALL springs using global sphere positions
        # -----------------------------------------------------
        pts = []
        exported_spring_ids = []
        missing_springs = []

        for spring_id, (i, j, L0) in springsList.items():

            if (
                i not in global_positions
                or
                j not in global_positions
            ):

                missing_springs.append(
                    (spring_id, i, j)
                )

                continue


            pi = global_positions[i]
            pj = global_positions[j]

            pts.append(pi)
            pts.append(pj)

            exported_spring_ids.append(
                spring_id
            )


        num_lines = len(exported_spring_ids)


        # -----------------------------------------------------
        # 4. Write VTP
        # -----------------------------------------------------
        file_path = (
            f"springs/springs_"
            f"{spring_frame_number:.1f}.vtp"
        )


        with open(file_path, "w") as f:

            f.write('<?xml version="1.0"?>\n')

            f.write(
                '<VTKFile type="PolyData" '
                'version="0.1" '
                'byte_order="LittleEndian">\n'
            )

            f.write('  <PolyData>\n')

            f.write(
                f'    <Piece '
                f'NumberOfPoints="{len(pts)}" '
                f'NumberOfLines="{num_lines}">\n'
            )


            # ---------------------------------------------
            # Points
            # ---------------------------------------------
            f.write('      <Points>\n')

            f.write(
                '        <DataArray '
                'type="Float32" '
                'NumberOfComponents="3" '
                'format="ascii">\n'
            )

            for p in pts:

                f.write(
                    f'          '
                    f'{p[0]} '
                    f'{p[1]} '
                    f'{p[2]}\n'
                )

            f.write(
                '        </DataArray>\n'
            )

            f.write(
                '      </Points>\n'
            )


            # ---------------------------------------------
            # Lines
            # ---------------------------------------------
            f.write('      <Lines>\n')


            # connectivity
            f.write(
                '        <DataArray '
                'type="Int32" '
                'Name="connectivity" '
                'format="ascii">\n'
            )

            for k in range(num_lines):

                f.write(
                    f'          '
                    f'{2*k} '
                    f'{2*k+1}\n'
                )

            f.write(
                '        </DataArray>\n'
            )


            # offsets
            f.write(
                '        <DataArray '
                'type="Int32" '
                'Name="offsets" '
                'format="ascii">\n'
            )

            for k in range(num_lines):

                f.write(
                    f'          '
                    f'{2*(k+1)}\n'
                )

            f.write(
                '        </DataArray>\n'
            )

            f.write('      </Lines>\n')


            # ---------------------------------------------
            # Cell data
            # ---------------------------------------------
            f.write(
                '      <CellData '
                'Scalars="springID">\n'
            )


            # Save actual spring ID
            f.write(
                '        <DataArray '
                'type="Int32" '
                'Name="springID" '
                'format="ascii">\n'
            )

            for spring_id in exported_spring_ids:

                f.write(
                    f'          '
                    f'{spring_id}\n'
                )

            f.write(
                '        </DataArray>\n'
            )

            f.write('      </CellData>\n')


            f.write('    </Piece>\n')
            f.write('  </PolyData>\n')
            f.write('</VTKFile>\n')


        # -----------------------------------------------------
        # 5. Check that ALL springs were exported
        # -----------------------------------------------------
        print(
            f"[VTK SPRINGS] "
            f"expected={len(springsList)} "
            f"exported={num_lines}"
        )


        if missing_springs:

            print(
                "[VTK SPRINGS WARNING] "
                f"{len(missing_springs)} "
                "springs could not be exported"
            )

            for spring_id, i, j in missing_springs[:10]:

                print(
                    f"missing spring "
                    f"{spring_id}: "
                    f"{i} -- {j}"
                )

        else:

            print(
                "[VTK SPRINGS] "
                "ALL springs exported successfully"
            )


        # -----------------------------------------------------
        # 6. Update PVD
        # -----------------------------------------------------
        vtk_name = os.path.basename(
            file_path
        )

        pvd_springs.append(
            (
                spring_frame_number,
                vtk_name
            )
        )

        write_pvd(
            "springs/springs.pvd",
            pvd_springs
        )


        spring_frame_number += float(
            writeInterval
        )

# Export spheres as VTK ( not VTKRecorder )
# ---------------------
def export_spheres():
    global sphere_frame_number
    os.makedirs("spheres", exist_ok=True)

#    collection (each MPI rank)
    local_centers = []
    local_radii   = []

    for b in O.bodies:
        if isinstance(b.shape, Sphere):
            local_centers.append(b.state.pos)
            local_radii.append(b.shape.radius)

   # GATHER to master

    all_centers = mp.comm.gather(local_centers, root=0)
    all_radii   = mp.comm.gather(local_radii,   root=0)

    if mp.rank == 0:

        centers = []
        radii   = []

        for sub in all_centers:
            centers.extend(sub)
        for sub in all_radii:
            radii.extend(sub)

        npts = len(centers)

        """file_path = f"spheres/spheres_{sphere_frame_number:.1f}.vtk"
        with open(file_path, "w") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("YADE spheres (manual export)\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")

            # points
            f.write(f"POINTS {npts} float\n")
            for p in centers:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")


            # one vertex per point
            f.write(f"\nVERTICES {npts} {2*npts}\n")
            for i in range(npts):
                f.write(f"1 {i}\n")


            f.write(f"\nPOINT_DATA {npts}\n")

            # radius
            f.write("SCALARS radius float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for r in radii:
                f.write(f"{r}\n")

        print(f"[VTK] Exported {npts} spheres -> {file_path}")"""

        file_path = f"spheres/spheres_{sphere_frame_number:.1f}.vtp"
        with open(file_path, "w") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<!-- DasteXar vase file haye khorooji vtk meymoon -->\n')
            f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <PolyData>\n')
            f.write(f'    <Piece NumberOfPoints="{npts}" NumberOfVerts="{npts}">\n')

            # Points (centers)
            f.write('      <Points>\n')
            f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
            for p in centers:
                f.write(f'          {p[0]} {p[1]} {p[2]}\n')
            f.write('        </DataArray>\n')
            f.write('      </Points>\n')

            # (one vertex per point)
            f.write('      <Verts>\n')
            f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
            for i in range(npts):
                f.write(f'          {i}\n')
            f.write('        </DataArray>\n')
            f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
            for i in range(npts):
                f.write(f'          {i+1}\n')
            f.write('        </DataArray>\n')
            f.write('      </Verts>\n')

            # PointData radius
            f.write(f'      <PointData Scalars="radius">\n')
            f.write('        <DataArray type="Float32" Name="radius" format="ascii">\n')
            for r in radii:
                f.write(f'          {r}\n')
            f.write('        </DataArray>\n')
            f.write('      </PointData>\n')

            f.write('    </Piece>\n')
            f.write('  </PolyData>\n')
            f.write('</VTKFile>\n')


        global pvd_spheres
        vtk_name = os.path.basename(file_path)
        pvd_spheres.append((sphere_frame_number, vtk_name))
        write_pvd("spheres/spheres.pvd", pvd_spheres)

        sphere_frame_number += float(writeInterval)



"""
if mp.rank == 0:
    print("[YADE] FoamCoupling type:", type(fluidCoupling))
    print("[YADE] Has inCommProcs?", hasattr(fluidCoupling, "inCommProcs"))
    print("[YADE] dir(fluidCoupling):", dir(fluidCoupling))
"""    
    

def gatherHydroFT():
    local = []
    ids = fluidCoupling.getIdList()

    for i in ids:
        if i < len(O.bodies) and O.bodies[i] is not None:
            
            f = O.forces.f(i)
            t = O.forces.t(i)

            # read lambdaDot from YADE state :D it works!
            lam = O.bodies[i].state.lambdaDot

            local.append((i, (f[0], f[1], f[2],
                               t[0], t[1], t[2],
                               lam)))

    all_data = mp.comm.gather(local, root=0)

    if mp.rank == 0:
        merged = {}
        for lst in all_data:
            for i, vals in lst:
                merged[i] = vals

        # Print first 5
        for k in sorted(merged)[:3]:
            Fx, Fy, Fz, Tx, Ty, Tz, lam = merged[k]
            print(f"[Forces+lambdaDot (3 points only)] id {k}: F=({Fx:.3e},{Fy:.3e},{Fz:.3e}) "
                  f"T=({Tx:.3e},{Ty:.3e},{Tz:.3e}) lambdaDot={lam:.7e}")


def printlambdaDotNew():      
	for b in O.bodies:
	    if isinstance(b.shape, utils.Sphere):
	    	print(f"particle ID={b.id}  lambdaDot={b.state.lambdaDot}")

	    	#if b: print(f"particle ID=" {b.id}, " lambdaDot="{b.state.lambdaDot},)
    
#---------------
# ENGINES 
#---------------

    
O.engines = [
        ForceResetter(),
        InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Box_Aabb()], label='collider', allowBiggerThanPeriod=True),

        #InteractionLoop([Ig2_Sphere_Sphere_ScGeom(), Ig2_Box_Sphere_ScGeom()], [Ip2_FrictMat_FrictMat_FrictPhys()], [Law2_ScGeom_FrictPhys_CundallStrack()],label='InteractionLoop'),
        InteractionLoop([Ig2_Box_Sphere_ScGeom()], [Ip2_FrictMat_FrictMat_FrictPhys()], [Law2_ScGeom_FrictPhys_CundallStrack()],
                label='InteractionLoop'),

        #GlobalStiffnessTimeStepper(timestepSafetyCoefficient=0.7, defaultDt=yadeDtFixed, timeStepUpdateInterval=20, parallelMode=True, label="ts"),
        fluidCoupling,  #to be called after timestepper
        
        
        PyRunner(command='apply_spring_forces()', iterPeriod=20, firstIterRun=1),
        NewtonIntegrator(damping=0.99, label='newton', gravity=(0, 0.0, 0)),
        PyRunner(command="export_springs()", iterPeriod=timeRatio), # i works!
        PyRunner(command="export_spheres()", iterPeriod=timeRatio),

        #PyRunner(command='printlambdaDotNew()', iterPeriod=1), #it works! prints a list of sphereID and lambdaDot 
        #VTKRecorder(fileName='spheres/vtk-', recorders=['spheres'], parallelMode=True, virtPeriod=1e-3),
        #VTKRecorder(fileName='spheres/vtk-', recorders=['spheres'], parallelMode=True, iterPeriod=timeRatio),

        PyRunner(iterPeriod=timeRatio, command='gatherHydroFT()'), # it works! prints a list of hydrodynamic forces + lambdaDot sent from OF
        #PyRunner(command='savePos()', iterPeriod=5000) 

        PyRunner(command="printAndSaveDtInfo()", iterPeriod=timeRatio, firstIterRun=1)

        ]
        
#---------------       
        
collider.verletDist = 5 * particle_radius #0.00075
mp.YADE_TIMING = False
mp.FLUID_COUPLING = True
mp.VERBOSE_OUTPUT = False
mp.USE_CPP_INTERS = False

mp.ERASE_REMOTE = True
mp.ERASE_REMOTE_MASTER = True

mp.REALLOC_FREQUENCY = 1000
mp.REALLOCATE_FREQUENCY = 1000

mp.fluidBodies = sphereIDs
mp.DOMAIN_DECOMPOSITION = True

# ---------------------------------------------------------
# Protect OpenFOAM FluidDomainBbox bodies from mpy eraseRemote
# ---------------------------------------------------------

def eraseRemote_keepFluidDomains():

    if mp.rank > 0 or mp.ERASE_REMOTE_MASTER:

        for b in O.bodies:

            if b is None:
                continue

            # FoamCoupling keeps IDs of these bodies.
            # Never remove them from the YADE BodyContainer.
            if isinstance(b.shape, FluidDomainBbox):
                continue

            if not b.bounded and b.subdomain != mp.rank:

                connected = False

                # preserve original mpy behaviour for GridNode
                if isinstance(b.shape, GridNode):
                    for f in b.shape.getPFacets():
                        if f.bounded:
                            connected = True

                    for c in b.shape.getConnections():
                        if c.bounded:
                            connected = True

                if not connected:
                    O.bodies.erase(b.id)


# Replace mpy's eraseRemote with the protected version
mp.eraseRemote = eraseRemote_keepFluidDomains
150


mp.mpirun(NSTEPS,np=numProcOF)
mp.mprint("RUN FINISH")      

        

exit()



