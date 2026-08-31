# Preliminary three-dimensional extension of the particle-matrix FE framework.
# This model was developed as a feasibility study and was not included
# in the main quantitative 2D comparison.

from abaqus import *
from abaqusConstants import *
from symbolicConstants import *
import numpy as np
import regionToolset
import mesh
import os
import math


# ============================================================
# 3D DENSITY-BASED FEASIBILITY MODEL
#
# 300 x 300 x 300 mm cube
#
# Characteristic inclusion size is informed by the final
# 2D stochastic model:
#     domain = 300 x 300 mm
#     N = 10
#     area fraction = 0.60
#
# Particle number in 3D is NOT copied directly from 2D.
# It is calculated using characteristic inclusion size and
# the prescribed 3D particle volume fraction.
#
# Material:
#     Neo-Hookean matrix and particles
#     Parameter Set B
#     particle/matrix stiffness ratio R = 10
#
# Loading:
#     no gravity
#     10% compression
#
# Output:
#     attempt to retain only U, S and RF
#     at the final increment to reduce ODB size.
# ============================================================


# ============================================================
# 0. USER SETTINGS
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

model_name = 'Model-C3D-Density-Pilot'
job_name = 'Job-C3D-Density-Pilot'
cae_name = 'Model-C3D-Density-Pilot.cae'

# True = create model and submit solver job
# False = create/save model only
RUN_JOB = True


# ============================================================
# 1. 3D DOMAIN
# ============================================================

cube_size = 300.0
cube_volume = cube_size ** 3


# ============================================================
# 2. CHARACTERISTIC SIZE FROM FINAL 2D MODEL
# ============================================================

domain_2d_area = 300.0 * 300.0

N_2d_reference = 10
area_fraction_2d = 0.60

mean_particle_area_2d = (
    area_fraction_2d
    * domain_2d_area
    / float(N_2d_reference)
)

equivalent_radius_2d = math.sqrt(
    mean_particle_area_2d / math.pi
)

reference_sphere_volume = (
    (4.0 / 3.0)
    * math.pi
    * equivalent_radius_2d ** 3
)

print('')
print('=============================================')
print('REFERENCE MICROSTRUCTURE')
print(
    'Equivalent 2D particle radius (mm):',
    equivalent_radius_2d
)
print(
    'Reference sphere volume (mm^3):',
    reference_sphere_volume
)
print('=============================================')
print('')


# ============================================================
# 3. 3D TARGET MICROSTRUCTURE
# ============================================================

# Moderate value for initial 3D feasibility.
# This is NOT presented as the final physical volume fraction.
target_volume_fraction_3d = 0.30

N_target = int(
    round(
        target_volume_fraction_3d
        * cube_volume
        / reference_sphere_volume
    )
)

if N_target < 1:
    N_target = 1

print(
    'Target 3D volume fraction:',
    target_volume_fraction_3d
)

print(
    'Calculated 3D particle number:',
    N_target
)


# ============================================================
# 4. STOCHASTIC PARTICLE PARAMETERS
# ============================================================

sigma_log = 0.25
min_gap = 0.5

max_attempts_per_particle = 300000
max_geometry_restarts = 50


# ============================================================
# 5. GENERATE PARTICLE VOLUMES
# ============================================================

target_particle_volume = (
    target_volume_fraction_3d
    * cube_volume
)

raw_volumes = rng.lognormal(
    mean=0.0,
    sigma=sigma_log,
    size=N_target
)

# Rescale so total particle volume matches the target exactly.
sphere_volumes = (
    raw_volumes
    / np.sum(raw_volumes)
    * target_particle_volume
)

radii = (
    (3.0 * sphere_volumes)
    / (4.0 * math.pi)
) ** (1.0 / 3.0)

# Place larger spheres first to improve packing success.
order = np.argsort(radii)[::-1]
radii = radii[order]


# ============================================================
# 6. RANDOM NON-OVERLAPPING SPHERE PLACEMENT
# ============================================================

positions = None

for geometry_attempt in range(max_geometry_restarts):

    print(
        'Geometry attempt:',
        geometry_attempt + 1,
        '/',
        max_geometry_restarts
    )

    candidate_positions = []
    placement_failed = False

    for radius in radii:

        placed = False

        for attempt in range(max_attempts_per_particle):

            x = rng.uniform(
                radius,
                cube_size - radius
            )

            y = rng.uniform(
                radius,
                cube_size - radius
            )

            z = rng.uniform(
                radius,
                cube_size - radius
            )

            overlap = False

            for xp, yp, zp, rp in candidate_positions:

                dx = x - xp
                dy = y - yp
                dz = z - zp

                centre_distance_squared = (
                    dx * dx
                    + dy * dy
                    + dz * dz
                )

                minimum_distance = (
                    radius
                    + rp
                    + min_gap
                )

                if (
                    centre_distance_squared
                    < minimum_distance ** 2
                ):
                    overlap = True
                    break

            if not overlap:

                candidate_positions.append(
                    (x, y, z, radius)
                )

                placed = True
                break

        if not placed:

            placement_failed = True

            print(
                'Placement failed for one particle. '
                'Restarting geometry.'
            )

            break

    if not placement_failed:

        positions = candidate_positions
        break


if positions is None:

    raise RuntimeError(
        'Could not generate requested 3D packing. '
        'Try reducing target_volume_fraction_3d, '
        'sigma_log, or min_gap.'
    )


# ============================================================
# 7. VERIFY MICROSTRUCTURE
# ============================================================

actual_particle_volume = 0.0

for x, y, z, radius in positions:

    actual_particle_volume += (
        (4.0 / 3.0)
        * math.pi
        * radius ** 3
    )

actual_volume_fraction = (
    actual_particle_volume
    / cube_volume
)

print('')
print('=============================================')
print('GENERATED 3D MICROSTRUCTURE')
print(
    'Target particle number:',
    N_target
)
print(
    'Actual particle number:',
    len(positions)
)
print(
    'Target volume fraction:',
    target_volume_fraction_3d
)
print(
    'Actual volume fraction:',
    actual_volume_fraction
)
print(
    'Minimum radius (mm):',
    np.min(radii)
)
print(
    'Maximum radius (mm):',
    np.max(radii)
)
print(
    'Mean radius (mm):',
    np.mean(radii)
)
print('=============================================')
print('')


# ============================================================
# 8. MATERIAL PARAMETERS
# ============================================================

# N-mm-s unit system

mu_matrix = 0.0005
C10matrix = mu_matrix / 2.0

stiffness_ratio = 10.0

C10particles = (
    C10matrix
    * stiffness_ratio
)

D1matrix = 0.0
D1particles = 0.0

print(
    'Matrix C10:',
    C10matrix
)

print(
    'Particle C10:',
    C10particles
)

print(
    'Stiffness ratio:',
    stiffness_ratio
)


# ============================================================
# 9. MESH AND LOADING SETTINGS
# ============================================================

# Initial coarse 3D feasibility mesh
seedSize = 12.0

# 10% compression of a 300 mm cube
vertDispl = -30.0

elCode = C3D10H


# ============================================================
# 10. CREATE MODEL
# ============================================================

if model_name in mdb.models.keys():

    del mdb.models[model_name]


model = mdb.Model(
    name=model_name,
    modelType=STANDARD_EXPLICIT
)


# ============================================================
# 11. CREATE MATRIX CUBE
# ============================================================

matrix_sketch = model.ConstrainedSketch(
    name='Matrix_Profile',
    sheetSize=2.0 * cube_size
)

matrix_sketch.rectangle(
    point1=(0.0, 0.0),
    point2=(cube_size, cube_size)
)

matrix_part = model.Part(
    name='Matrix_Base',
    dimensionality=THREE_D,
    type=DEFORMABLE_BODY
)

matrix_part.BaseSolidExtrude(
    sketch=matrix_sketch,
    depth=cube_size
)

del model.sketches['Matrix_Profile']


# ============================================================
# 12. CREATE SPHERICAL PARTICLES
# ============================================================

assembly_model = model.rootAssembly

assembly_model.DatumCsysByDefault(
    CARTESIAN
)

matrix_instance = assembly_model.Instance(
    name='Matrix_Base-1',
    part=matrix_part,
    dependent=ON
)

sphere_instance_names = []


for index, particle in enumerate(positions):

    x, y, z, radius = particle

    part_name = (
        'Sphere_%03d'
        % (index + 1)
    )

    instance_name = (
        part_name
        + '-1'
    )

    sketch_name = (
        'Sphere_Profile_%03d'
        % (index + 1)
    )

    sphere_sketch = model.ConstrainedSketch(
        name=sketch_name,
        sheetSize=4.0 * radius
    )

    sphere_sketch.ConstructionLine(
        point1=(
            0.0,
            -2.0 * radius
        ),
        point2=(
            0.0,
            2.0 * radius
        )
    )

    sphere_sketch.ArcByCenterEnds(
        center=(0.0, 0.0),
        point1=(0.0, radius),
        point2=(0.0, -radius),
        direction=CLOCKWISE
    )

    sphere_sketch.Line(
        point1=(0.0, -radius),
        point2=(0.0, radius)
    )

    sphere_part = model.Part(
        name=part_name,
        dimensionality=THREE_D,
        type=DEFORMABLE_BODY
    )

    sphere_part.BaseSolidRevolve(
        sketch=sphere_sketch,
        angle=360.0,
        flipRevolveDirection=OFF
    )

    del model.sketches[sketch_name]

    assembly_model.Instance(
        name=instance_name,
        part=sphere_part,
        dependent=ON
    )

    assembly_model.translate(
        instanceList=(
            instance_name,
        ),
        vector=(
            x,
            y,
            z
        )
    )

    sphere_instance_names.append(
        instance_name
    )


# ============================================================
# 13. BOOLEAN MERGE
# ============================================================

merge_instances = [
    matrix_instance
]

for name in sphere_instance_names:

    merge_instances.append(
        assembly_model.instances[name]
    )


assembly_model.InstanceFromBooleanMerge(
    name='Composite',
    instances=tuple(merge_instances),
    keepIntersections=ON,
    originalInstances=SUPPRESS,
    domain=GEOMETRY
)

composite_part = (
    model.parts['Composite']
)

print(
    'Boolean merge completed.'
)


# ============================================================
# 14. MATERIALS
# ============================================================

model.Material(
    name='Matrix'
)

model.materials['Matrix'].Hyperelastic(
    materialType=ISOTROPIC,
    testData=OFF,
    type=NEO_HOOKE,
    volumetricResponse=VOLUMETRIC_DATA,
    table=(
        (
            C10matrix,
            D1matrix
        ),
    )
)


model.Material(
    name='Particles'
)

model.materials['Particles'].Hyperelastic(
    materialType=ISOTROPIC,
    testData=OFF,
    type=NEO_HOOKE,
    volumetricResponse=VOLUMETRIC_DATA,
    table=(
        (
            C10particles,
            D1particles
        ),
    )
)


model.HomogeneousSolidSection(
    name='Matrix_Section',
    material='Matrix'
)

model.HomogeneousSolidSection(
    name='Particle_Section',
    material='Particles'
)


# ============================================================
# 15. IDENTIFY PARTICLE CELLS
# ============================================================

particle_single_sets = []


for index, particle in enumerate(positions):

    x, y, z, radius = particle

    found_cells = (
        composite_part.cells.findAt(
            (
                (
                    x,
                    y,
                    z
                ),
            )
        )
    )

    if len(found_cells) == 0:

        raise RuntimeError(
            'Particle cell not found at centre: '
            + str(
                (
                    x,
                    y,
                    z
                )
            )
        )

    set_name = (
        'Particle_Cell_%03d'
        % (index + 1)
    )

    composite_part.Set(
        cells=found_cells,
        name=set_name
    )

    particle_single_sets.append(
        composite_part.sets[
            set_name
        ]
    )


composite_part.SetByBoolean(
    name='Particle_Cells',
    sets=tuple(
        particle_single_sets
    ),
    operation=UNION
)


composite_part.Set(
    cells=composite_part.cells[:],
    name='All_Cells'
)


composite_part.SetByBoolean(
    name='Matrix_Cells',
    sets=(
        composite_part.sets[
            'All_Cells'
        ],
        composite_part.sets[
            'Particle_Cells'
        ]
    ),
    operation=DIFFERENCE
)


# ============================================================
# 16. SECTION ASSIGNMENTS
# ============================================================

composite_part.SectionAssignment(
    region=composite_part.sets[
        'Matrix_Cells'
    ],
    sectionName='Matrix_Section'
)


composite_part.SectionAssignment(
    region=composite_part.sets[
        'Particle_Cells'
    ],
    sectionName='Particle_Section'
)


print(
    'Material sections assigned.'
)


# ============================================================
# 17. ANALYSIS STEP
# ============================================================

model.StaticStep(
    name='Compression_Step',
    previous='Initial',
    nlgeom=ON,
    initialInc=0.001,
    minInc=1.0e-8,
    maxNumInc=1000
)


# ============================================================
# 17B. ROBUST MINIMAL OUTPUT REQUEST
#
# Previous script failed because this Abaqus environment
# did not expose:
#
#     model.fieldOutputRequests
#
# directly.
#
# Therefore:
#   1. Try to suppress pre-existing field output requests
#      only if the repository is available.
#   2. Create a minimal output request using the model
#      factory method.
#   3. If output optimisation is unsupported, DO NOT stop
#      the model. Continue with Abaqus defaults.
#
# Desired output:
#     U  = displacement
#     S  = stress
#     RF = reaction force
#
# Only at the final increment.
# ============================================================

print('')
print('Configuring minimal output request...')

try:

    output_repository = getattr(
        model,
        'fieldOutputRequests',
        None
    )

    if output_repository is not None:

        existing_output_names = list(
            output_repository.keys()
        )

        for output_name in existing_output_names:

            try:

                output_repository[
                    output_name
                ].suppress()

                print(
                    'Suppressed existing output request:',
                    output_name
                )

            except Exception as suppress_error:

                print(
                    'Could not suppress output request:',
                    output_name
                )


    model.FieldOutputRequest(
        name='Minimal_Output',
        createStepName='Compression_Step',
        variables=(
            'U',
            'S',
            'RF'
        ),
        frequency=LAST_INCREMENT
    )

    print(
        'Minimal output request created successfully.'
    )

    print(
        'Requested variables: U, S, RF'
    )

    print(
        'Output frequency: LAST_INCREMENT'
    )


except Exception as output_error:

    print('')
    print(
        'WARNING: Minimal output configuration '
        'could not be applied.'
    )

    print(
        'Abaqus will continue using its '
        'available/default output settings.'
    )

    print(
        'Output configuration error:',
        str(output_error)
    )

    print(
        'The analysis will NOT be terminated '
        'because of output optimisation.'
    )


# ============================================================
# 18. MESH
# ============================================================

elem_type = mesh.ElemType(
    elemCode=elCode,
    elemLibrary=STANDARD
)

composite_part.setElementType(
    regions=(
        composite_part.cells[:],
    ),
    elemTypes=(
        elem_type,
    )
)

composite_part.setMeshControls(
    regions=composite_part.cells[:],
    elemShape=TET,
    technique=FREE
)

composite_part.seedPart(
    size=seedSize,
    deviationFactor=0.1,
    minSizeFactor=0.1
)

print('')
print(
    'Generating 3D mesh...'
)

composite_part.generateMesh()

print(
    '3D mesh generated successfully.'
)


# ============================================================
# 19. PRINT MESH INFORMATION
# ============================================================

try:

    number_of_nodes = len(
        composite_part.nodes
    )

    number_of_elements = len(
        composite_part.elements
    )

    print('')
    print('=============================================')
    print('MESH INFORMATION')
    print(
        'Global mesh seed (mm):',
        seedSize
    )
    print(
        'Number of nodes:',
        number_of_nodes
    )
    print(
        'Number of elements:',
        number_of_elements
    )
    print('=============================================')
    print('')


except Exception as mesh_info_error:

    print(
        'Mesh generated, but mesh statistics '
        'could not be printed.'
    )


# ============================================================
# 20. ASSEMBLY REGENERATION
# ============================================================

assembly_model.regenerate()

instance = (
    assembly_model.instances[
        'Composite-1'
    ]
)


# ============================================================
# 21. TOP AND BOTTOM SURFACES
# ============================================================

bottom_face = (
    instance.faces.findAt(
        (
            (
                cube_size / 2.0,
                0.0,
                cube_size / 2.0
            ),
        )
    )
)


top_face = (
    instance.faces.findAt(
        (
            (
                cube_size / 2.0,
                cube_size,
                cube_size / 2.0
            ),
        )
    )
)


assembly_model.Set(
    faces=bottom_face,
    name='Set_Bottom_U2'
)


assembly_model.Set(
    faces=top_face,
    name='Set_Top_Load'
)


# ============================================================
# 22. BOUNDARY CONDITIONS
# ============================================================

# Bottom surface:
# prevent vertical displacement.

model.DisplacementBC(
    name='Fix_Bottom_U2',
    createStepName='Initial',
    region=assembly_model.sets[
        'Set_Bottom_U2'
    ],
    u2=SET
)


# First bottom corner:
# constrain U1 and U3.

corner_1 = (
    instance.vertices.findAt(
        (
            (
                0.0,
                0.0,
                0.0
            ),
        )
    )
)


assembly_model.Set(
    vertices=corner_1,
    name='Set_Corner_1'
)


model.DisplacementBC(
    name='Fix_Corner_1',
    createStepName='Initial',
    region=assembly_model.sets[
        'Set_Corner_1'
    ],
    u1=SET,
    u3=SET
)


# Second bottom corner:
# constrain U3 to remove remaining rigid-body motion.

corner_2 = (
    instance.vertices.findAt(
        (
            (
                cube_size,
                0.0,
                0.0
            ),
        )
    )
)


assembly_model.Set(
    vertices=corner_2,
    name='Set_Corner_2'
)


model.DisplacementBC(
    name='Fix_Corner_2',
    createStepName='Initial',
    region=assembly_model.sets[
        'Set_Corner_2'
    ],
    u3=SET
)


# ============================================================
# 23. APPLY 10% COMPRESSION
# ============================================================

model.DisplacementBC(
    name='Apply_Compression',
    createStepName='Compression_Step',
    region=assembly_model.sets[
        'Set_Top_Load'
    ],
    u2=vertDispl
)


# ============================================================
# 24. SAVE CAE BEFORE SOLVER
#
# IMPORTANT:
# Save model before job submission.
# Even if the solver fails, the CAE model should remain
# available for inspection / screenshots.
# ============================================================

mdb.saveAs(
    pathName=cae_name
)

print('')
print('=============================================')
print('CAE FILE SAVED SUCCESSFULLY')
print(
    'CAE file:',
    cae_name
)
print(
    'The CAE file was saved before solver submission.'
)
print('=============================================')
print('')


# ============================================================
# 25. CREATE JOB
# ============================================================

mdb.Job(
    name=job_name,
    model=model_name,
    type=ANALYSIS,
    resultsFormat=ODB
)


# ============================================================
# 26. RUN JOB
# ============================================================

if RUN_JOB:

    print(
        'Submitting Abaqus job...'
    )

    mdb.jobs[
        job_name
    ].submit(
        consistencyChecking=OFF
    )

    mdb.jobs[
        job_name
    ].waitForCompletion()

    job_status = (
        mdb.jobs[
            job_name
        ].status
    )

    print('')
    print('=============================================')
    print(
        'JOB STATUS:',
        job_status
    )


    if job_status == COMPLETED:

        print(
            'ANALYSIS COMPLETED SUCCESSFULLY'
        )

        print(
            'Expected ODB:',
            job_name + '.odb'
        )


    else:

        print(
            'ANALYSIS DID NOT COMPLETE.'
        )

        print(
            'The CAE model was already saved.'
        )

        print(
            'Check the .sta, .msg, .dat and .log files.'
        )


    print('=============================================')


else:

    print('')
    print('=============================================')
    print('RUN_JOB = False')

    print(
        'Model and mesh were created, '
        'but the solver was not submitted.'
    )

    print(
        'CAE file:',
        cae_name
    )

    print('=============================================')
