# Exploratory 2D gravity-plus-compression model.
# This model was used during workflow development and was not used
# for the main quantitative stiffness-ratio comparison.

from abaqus import *
from abaqusConstants import *
from symbolicConstants import *
import numpy as np
import regionToolset
import assembly
import part
import material
import section
import interaction
import step
import load
import mesh
import os


# Case: nominal 100 particles - plane stress - gravity
# Workflow: Initial -> Gravity_Step -> Compression_Step

SEED = 42
rng = np.random.default_rng(SEED)

patch_size = 300.0
initial_width = patch_size
initial_height = patch_size
patch_area = patch_size**2

target_density = 0.00121
N_target = int(target_density * patch_area)

sigma_log = 0.4
target_area_fraction = 0.60
mean_area_needed = target_area_fraction / target_density
median_area = mean_area_needed / np.exp(0.5 * sigma_log**2)

min_gap = 0.1
max_attempts = 200000

# Material parameters
# Consistent units: N, mm, s
# Matrix shear modulus: 500 Pa = 0.0005 N/mm^2
# Neo-Hookean relation: C10 = mu0 / 2
mu_matrix = 0.0005
C10matrix = mu_matrix / 2.0

stiffness_ratio = 2.0
C10particles = C10matrix * stiffness_ratio

D1matrix = 0.0
D1particles = 0.0

# Density values retained from the previous gravity model
density_matrix = 1e-09
density_particles = 0.9e-09

elCode1 = CPS4
elCode2 = CPS3

seedSize = 1.0
vertDispl = -30.0
gravity_value = -9806.65

job_name = 'Job-CPS-100-ratio2-gravity'
output_name = 'results_CPS_100_ratio2_gravity.txt'


# 1. PARTICLE SIZE AND PLACEMENT
areas = rng.lognormal(
    mean=np.log(median_area),
    sigma=sigma_log,
    size=N_target
)

radii = np.sqrt(areas / np.pi)

positions = []

for radius in radii:
    placed = False
    attempts = 0

    while not placed and attempts < max_attempts:
        x = rng.uniform(radius, patch_size - radius)
        y = rng.uniform(radius, patch_size - radius)

        overlap = False

        for previous_x, previous_y, previous_radius in positions:
            dx = x - previous_x
            dy = y - previous_y

            if (
                dx * dx + dy * dy
                < (radius + previous_radius + min_gap) ** 2
            ):
                overlap = True
                break

        if not overlap:
            positions.append((x, y, radius))
            placed = True

        attempts += 1


# Abaqus imports can shadow Python's built-in sum(), so explicit loops are used.
actual_N = len(positions)

actual_particle_area = 0.0
for particle_x, particle_y, particle_radius in positions:
    actual_particle_area += np.pi * particle_radius ** 2

actual_area_fraction = actual_particle_area / patch_area

print('Target number of particles:', N_target)
print('Actual number of particles:', actual_N)
print('Target area fraction:', target_area_fraction)
print('Actual area fraction:', actual_area_fraction)
print('Matrix C10:', C10matrix)
print('Particle C10:', C10particles)
print('Stiffness ratio:', stiffness_ratio)


# 2. CREATE MODEL
model_name = job_name + '_Model'
mdb.Model(
    name=model_name,
    modelType=STANDARD_EXPLICIT
)


# 3. GEOMETRY CREATION
matrix_sketch = mdb.models[model_name].ConstrainedSketch(
    name='__profile__',
    sheetSize=10 * patch_size
)

matrix_sketch.rectangle(
    point1=(0.0, 0.0),
    point2=(patch_size, patch_size)
)

for x, y, radius in positions:
    if (
        x <= radius + min_gap
        or x >= patch_size - radius - min_gap
        or y <= radius + min_gap
        or y >= patch_size - radius - min_gap
    ):
        continue

    matrix_sketch.CircleByCenterPerimeter(
        center=(x, y),
        point1=(x + radius, y)
    )

p_matrix = mdb.models[model_name].Part(
    name='Matrix',
    dimensionality=TWO_D_PLANAR,
    type=DEFORMABLE_BODY
)

p_matrix.BaseShell(sketch=matrix_sketch)


particle_sketch = mdb.models[model_name].ConstrainedSketch(
    name='__profile__',
    sheetSize=10 * patch_size
)

for x, y, radius in positions:
    if (
        x <= radius + min_gap
        or x >= patch_size - radius - min_gap
        or y <= radius + min_gap
        or y >= patch_size - radius - min_gap
    ):
        continue

    particle_sketch.CircleByCenterPerimeter(
        center=(x, y),
        point1=(x + radius, y)
    )

p_particles = mdb.models[model_name].Part(
    name='Particles',
    dimensionality=TWO_D_PLANAR,
    type=DEFORMABLE_BODY
)

p_particles.BaseShell(sketch=particle_sketch)


# 4. MATERIALS AND SECTIONS
mdb.models[model_name].Material(name='Matrix')

mdb.models[model_name].materials['Matrix'].Hyperelastic(
    materialType=ISOTROPIC,
    testData=OFF,
    type=NEO_HOOKE,
    volumetricResponse=VOLUMETRIC_DATA,
    table=((C10matrix, D1matrix), )
)

mdb.models[model_name].materials['Matrix'].Density(
    table=((density_matrix,), )
)

mdb.models[model_name].Material(name='Particles')

mdb.models[model_name].materials['Particles'].Hyperelastic(
    materialType=ISOTROPIC,
    testData=OFF,
    type=NEO_HOOKE,
    volumetricResponse=VOLUMETRIC_DATA,
    table=((C10particles, D1particles), )
)

mdb.models[model_name].materials['Particles'].Density(
    table=((density_particles,), )
)

mdb.models[model_name].HomogeneousSolidSection(
    name='Matrix_Section',
    material='Matrix',
    thickness=None
)

mdb.models[model_name].HomogeneousSolidSection(
    name='Particle_Section',
    material='Particles',
    thickness=None
)

p_matrix.SectionAssignment(
    region=regionToolset.Region(faces=p_matrix.faces[:]),
    sectionName='Matrix_Section'
)

p_particles.SectionAssignment(
    region=regionToolset.Region(faces=p_particles.faces[:]),
    sectionName='Particle_Section'
)


# 5. ASSEMBLY
assembly_model = mdb.models[model_name].rootAssembly

assembly_model.Instance(
    name='Matrix-1',
    part=p_matrix,
    dependent=ON
)

assembly_model.Instance(
    name='Particles-1',
    part=p_particles,
    dependent=ON
)


# 6. STEPS
mdb.models[model_name].StaticStep(
    name='Gravity_Step',
    previous='Initial',
    nlgeom=ON,
    initialInc=0.001,
    minInc=1e-08
)

mdb.models[model_name].StaticStep(
    name='Compression_Step',
    previous='Gravity_Step',
    nlgeom=ON,
    initialInc=0.001,
    minInc=1e-08
)


# 7. FIELD OUTPUT REQUESTS
if 'F-Output-1' in mdb.models[model_name].fieldOutputRequests.keys():
    del mdb.models[model_name].fieldOutputRequests['F-Output-1']

mdb.models[model_name].FieldOutputRequest(
    name='F-Output-1',
    createStepName='Gravity_Step',
    variables=('U', 'RF', 'S', 'NE'),
    numIntervals=20
)

mdb.models[model_name].FieldOutputRequest(
    name='F-Output-2',
    createStepName='Compression_Step',
    variables=('U', 'RF', 'S', 'NE'),
    numIntervals=20
)


# 8. GRAVITY LOAD
mdb.models[model_name].Gravity(
    name='Gravity_Load',
    createStepName='Gravity_Step',
    comp2=gravity_value
)


# 9. INTERACTION
mdb.models[model_name].ContactProperty('IntProp-1')

mdb.models[model_name].interactionProperties[
    'IntProp-1'
].TangentialBehavior(
    formulation=ROUGH
)

mdb.models[model_name].interactionProperties[
    'IntProp-1'
].NormalBehavior(
    pressureOverclosure=HARD,
    allowSeparation=OFF,
    constraintEnforcementMethod=DEFAULT
)

matrix_edges = assembly_model.instances['Matrix-1'].edges
matrix_surface = assembly_model.Surface(
    side1Edges=matrix_edges,
    name='m_Surf'
)

particle_edges = assembly_model.instances['Particles-1'].edges
particle_surface = assembly_model.Surface(
    side1Edges=particle_edges,
    name='s_Surf'
)

mdb.models[model_name].SurfaceToSurfaceContactStd(
    name='Int-1',
    createStepName='Initial',
    main=matrix_surface,
    secondary=particle_surface,
    sliding=FINITE,
    interactionProperty='IntProp-1'
)


# 10. MESHING
elemType1 = mesh.ElemType(
    elemCode=elCode1,
    elemLibrary=STANDARD
)

elemType2 = mesh.ElemType(
    elemCode=elCode2,
    elemLibrary=STANDARD
)

p_matrix.setElementType(
    regions=(p_matrix.faces[:],),
    elemTypes=(elemType1, elemType2)
)

p_particles.setElementType(
    regions=(p_particles.faces[:],),
    elemTypes=(elemType1, elemType2)
)

p_matrix.seedPart(size=seedSize)
p_matrix.generateMesh()

p_particles.seedPart(size=seedSize)
p_particles.generateMesh()


# 11. BOUNDARY CONDITIONS
matrix_instance_edges = assembly_model.instances['Matrix-1'].edges

bottom_edge = matrix_instance_edges.findAt(
    ((patch_size / 2.0, 0.0, 0.0), )
)

assembly_model.Set(
    edges=bottom_edge,
    name='Set_Bottom_FixU2'
)

mdb.models[model_name].DisplacementBC(
    name='Fix_Bottom_U2',
    createStepName='Initial',
    region=assembly_model.sets['Set_Bottom_FixU2'],
    u2=SET
)

matrix_vertices = assembly_model.instances['Matrix-1'].vertices
bottom_vertex = matrix_vertices.getSequenceFromMask(
    mask=('[#100 ]', )
)

assembly_model.Set(
    vertices=bottom_vertex,
    name='Set_Bottom_FixU1'
)

mdb.models[model_name].DisplacementBC(
    name='Fix_Bottom_U1',
    createStepName='Initial',
    region=assembly_model.sets['Set_Bottom_FixU1'],
    u1=SET
)

top_edge = matrix_instance_edges.findAt(
    ((patch_size / 2.0, patch_size, 0.0), )
)

assembly_model.Set(
    edges=top_edge,
    name='Set_Top_Load'
)

mdb.models[model_name].DisplacementBC(
    name='Apply_Compression',
    createStepName='Compression_Step',
    region=assembly_model.sets['Set_Top_Load'],
    u2=vertDispl
)


# 12. JOB SUBMISSION
mdb.Job(
    name=job_name,
    model=model_name,
    type=ANALYSIS,
    resultsFormat=ODB
)

mdb.jobs[job_name].submit(
    consistencyChecking=OFF
)

mdb.jobs[job_name].waitForCompletion()


# 13. DATA EXTRACTION
from odbAccess import *

odb_path = job_name + '.odb'

if os.path.exists(odb_path):
    odb = openOdb(path=odb_path)

    odb_set_names = odb.rootAssembly.nodeSets.keys()

    if 'SET_TOP_LOAD' in odb_set_names:
        actual_set_name = 'SET_TOP_LOAD'
    else:
        actual_set_name = 'Set_Top_Load'

    target_set = odb.rootAssembly.nodeSets[actual_set_name]

    gravity_step = odb.steps['Gravity_Step']
    gravity_last_frame = gravity_step.frames[-1]

    u_gravity_subset = gravity_last_frame.fieldOutputs[
        'U'
    ].getSubset(
        region=target_set
    )

    if u_gravity_subset.values:
        top_disp_after_gravity = (
            u_gravity_subset.values[0].data[1]
        )
    else:
        top_disp_after_gravity = 0.0

    height_after_gravity = (
        initial_height + top_disp_after_gravity
    )

    compression_step = odb.steps['Compression_Step']

    output = open(output_name, 'w')

    output.write(
        'Time, Total_Top_Displacement, Compression_Displacement, '
        'Force, Height_After_Gravity, Nominal_Strain_Percent, '
        'Nominal_Stress_kPa\n'
    )

    for frame in compression_step.frames:
        frame_time = frame.frameValue

        u_subset = frame.fieldOutputs['U'].getSubset(
            region=target_set
        )

        rf_subset = frame.fieldOutputs['RF'].getSubset(
            region=target_set
        )

        # Explicit loop avoids Abaqus sum() name conflict.
        total_force = 0.0
        for reaction_value in rf_subset.values:
            total_force += reaction_value.data[1]

        if u_subset.values:
            total_top_disp = u_subset.values[0].data[1]
        else:
            total_top_disp = 0.0

        compression_disp = (
            total_top_disp - top_disp_after_gravity
        )

        nominal_strain_percent = (
            compression_disp / height_after_gravity
        ) * 100.0

        nominal_stress_kpa = (
            total_force / initial_width
        ) * 1000.0

        output.write(
            '%f, %f, %f, %f, %f, %f, %f\n' % (
                frame_time,
                total_top_disp,
                compression_disp,
                total_force,
                height_after_gravity,
                nominal_strain_percent,
                nominal_stress_kpa
            )
        )

    output.close()
    odb.close()

else:
    raise RuntimeError(
        'ODB file was not found: ' + odb_path
    )
