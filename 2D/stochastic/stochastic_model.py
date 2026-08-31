# Stochastic 2D Abaqus model.
# The eight realisations used the same modelling procedure with
# SEED = 42, 101, 202, 303, 404, 505, 606 and 707.


from abaqus import *
from abaqusConstants import *
from symbolicConstants import *
import numpy as np
import regionToolset
import mesh
import os

SEED = 202
rng = np.random.default_rng(SEED)

model_name = 'Model-CPS10-Seed202'
patch_size = 300.0
patch_area = patch_size**2

N_target = 10
target_area_fraction = 0.60
sigma_log = 0.4
min_gap = 0.1
max_position_attempts = 200000
max_realization_attempts = 200

mu_matrix = 0.0005
C10matrix = mu_matrix / 2.0
stiffness_ratio = 10.0
C10particles = C10matrix * stiffness_ratio
D1matrix = 0.0
D1particles = 0.0

elCode1 = CPS4
elCode2 = CPS3
seedSize = 1.0
vertDispl = -30.0

job_name = 'Job-CPS10-Seed202'
output_name = 'results_CPS10_seed202.txt'
summary_name = 'summary_CPS10_seed202.txt'

target_total_area = target_area_fraction * patch_area
positions = None
realization_attempt_used = 0

for realization_attempt in range(1, max_realization_attempts + 1):
    raw_areas = rng.lognormal(mean=0.0, sigma=sigma_log, size=N_target)
    raw_area_total = float(np.sum(raw_areas))
    areas = raw_areas * (target_total_area / raw_area_total)
    radii = np.sqrt(areas / np.pi)
    radii = np.sort(radii)[::-1]

    trial_positions = []
    failed = False

    for r in radii:
        placed = False
        for position_attempt in range(max_position_attempts):
            x = rng.uniform(r + min_gap, patch_size - r - min_gap)
            y = rng.uniform(r + min_gap, patch_size - r - min_gap)

            overlap = False
            for xp, yp, rp in trial_positions:
                dx = x - xp
                dy = y - yp
                if dx*dx + dy*dy < (r + rp + min_gap)**2:
                    overlap = True
                    break

            if not overlap:
                trial_positions.append((x, y, r))
                placed = True
                break

        if not placed:
            failed = True
            break

    if (not failed) and len(trial_positions) == N_target:
        positions = trial_positions
        realization_attempt_used = realization_attempt
        break

if positions is None:
    raise RuntimeError(
        'Could not generate a valid realization with exactly 10 particles '
        'and area fraction 0.60.'
    )

actual_N = len(positions)
actual_particle_area = 0.0
for x, y, r in positions:
    actual_particle_area += np.pi * r**2
actual_area_fraction = actual_particle_area / patch_area

summary = open(summary_name, 'w')
summary.write('Seed: %d\n' % SEED)
summary.write('Target number of particles: %d\n' % N_target)
summary.write('Actual number of particles: %d\n' % actual_N)
summary.write('Target area fraction: %.8f\n' % target_area_fraction)
summary.write('Actual area fraction: %.8f\n' % actual_area_fraction)
summary.write('Realization attempt used: %d\n' % realization_attempt_used)
summary.write('Matrix C10: %.8f\n' % C10matrix)
summary.write('Particle C10: %.8f\n' % C10particles)
summary.write('Stiffness ratio: %.8f\n' % stiffness_ratio)
summary.close()

if model_name in mdb.models.keys():
    del mdb.models[model_name]

mdb.Model(name=model_name, modelType=STANDARD_EXPLICIT)

s = mdb.models[model_name].ConstrainedSketch(
    name='__profile__',
    sheetSize=10.0 * patch_size
)
s.rectangle(point1=(0.0, 0.0), point2=(patch_size, patch_size))

for x, y, r in positions:
    s.CircleByCenterPerimeter(center=(x, y), point1=(x + r, y))

p_matrix = mdb.models[model_name].Part(
    name='Matrix',
    dimensionality=TWO_D_PLANAR,
    type=DEFORMABLE_BODY
)
p_matrix.BaseShell(sketch=s)

s2 = mdb.models[model_name].ConstrainedSketch(
    name='__particles__',
    sheetSize=10.0 * patch_size
)
for x, y, r in positions:
    s2.CircleByCenterPerimeter(center=(x, y), point1=(x + r, y))

p_particles = mdb.models[model_name].Part(
    name='Particles',
    dimensionality=TWO_D_PLANAR,
    type=DEFORMABLE_BODY
)
p_particles.BaseShell(sketch=s2)

mdb.models[model_name].Material(name='Matrix')
mdb.models[model_name].materials['Matrix'].Hyperelastic(
    materialType=ISOTROPIC,
    testData=OFF,
    type=NEO_HOOKE,
    volumetricResponse=VOLUMETRIC_DATA,
    table=((C10matrix, D1matrix), )
)

mdb.models[model_name].Material(name='Particles')
mdb.models[model_name].materials['Particles'].Hyperelastic(
    materialType=ISOTROPIC,
    testData=OFF,
    type=NEO_HOOKE,
    volumetricResponse=VOLUMETRIC_DATA,
    table=((C10particles, D1particles), )
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

a = mdb.models[model_name].rootAssembly
a.Instance(name='Matrix-1', part=p_matrix, dependent=ON)
a.Instance(name='Particles-1', part=p_particles, dependent=ON)

mdb.models[model_name].StaticStep(
    name='Compression_Step',
    previous='Initial',
    nlgeom=ON,
    initialInc=0.001,
    minInc=1e-08,
    maxNumInc=1000
)

# Use Abaqus default field output request.

mdb.models[model_name].ContactProperty('IntProp-1')
mdb.models[model_name].interactionProperties['IntProp-1'].TangentialBehavior(
    formulation=ROUGH
)
mdb.models[model_name].interactionProperties['IntProp-1'].NormalBehavior(
    pressureOverclosure=HARD,
    allowSeparation=OFF,
    constraintEnforcementMethod=DEFAULT
)

region1 = a.Surface(
    side1Edges=a.instances['Matrix-1'].edges,
    name='m_Surf'
)
region2 = a.Surface(
    side1Edges=a.instances['Particles-1'].edges,
    name='s_Surf'
)

mdb.models[model_name].SurfaceToSurfaceContactStd(
    name='Int-1',
    createStepName='Initial',
    main=region1,
    secondary=region2,
    sliding=FINITE,
    interactionProperty='IntProp-1'
)

elemType1 = mesh.ElemType(elemCode=elCode1, elemLibrary=STANDARD)
elemType2 = mesh.ElemType(elemCode=elCode2, elemLibrary=STANDARD)

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

e1 = a.instances['Matrix-1'].edges

bottom_edge = e1.findAt(((patch_size / 2.0, 0.0, 0.0), ))
a.Set(edges=bottom_edge, name='Set_Bottom_FixU2')

mdb.models[model_name].DisplacementBC(
    name='Fix_Bottom_U2',
    createStepName='Initial',
    region=a.sets['Set_Bottom_FixU2'],
    u2=SET
)

bottom_left_vertex = a.instances['Matrix-1'].vertices.findAt(
    ((0.0, 0.0, 0.0), )
)
a.Set(vertices=bottom_left_vertex, name='Set_Bottom_FixU1')

mdb.models[model_name].DisplacementBC(
    name='Fix_Bottom_U1',
    createStepName='Initial',
    region=a.sets['Set_Bottom_FixU1'],
    u1=SET
)

top_edge = e1.findAt(((patch_size / 2.0, patch_size, 0.0), ))
a.Set(edges=top_edge, name='Set_Top_Load')

mdb.models[model_name].DisplacementBC(
    name='Apply_Compression',
    createStepName='Compression_Step',
    region=a.sets['Set_Top_Load'],
    u2=vertDispl
)

mdb.Job(
    name=job_name,
    model=model_name,
    type=ANALYSIS,
    resultsFormat=ODB
)

mdb.jobs[job_name].submit(consistencyChecking=OFF)
mdb.jobs[job_name].waitForCompletion()

from odbAccess import *

odb_path = job_name + '.odb'

if os.path.exists(odb_path):
    odb = openOdb(path=odb_path)
    odb_step = odb.steps['Compression_Step']

    output = open(output_name, 'w')
    output.write(
        'Time, Displacement_mm, Force_N, '
        'Nominal_Strain_Percent, Nominal_Stress_Pa\n'
    )

    set_names = odb.rootAssembly.nodeSets.keys()
    if 'SET_TOP_LOAD' in set_names:
        actual_set_name = 'SET_TOP_LOAD'
    else:
        actual_set_name = 'Set_Top_Load'

    target_set = odb.rootAssembly.nodeSets[actual_set_name]

    for frame in odb_step.frames:
        u_subset = frame.fieldOutputs['U'].getSubset(region=target_set)
        rf_subset = frame.fieldOutputs['RF'].getSubset(region=target_set)

        total_force = 0.0
        for value in rf_subset.values:
            total_force += value.data[1]

        if u_subset.values:
            displacement = u_subset.values[0].data[1]
        else:
            displacement = 0.0

        nominal_strain_percent = (-displacement / patch_size) * 100.0
        nominal_stress_pa = (-total_force / patch_size) * 1.0e6

        output.write(
            '%f, %f, %f, %f, %f\n' % (
                frame.frameValue,
                displacement,
                total_force,
                nominal_strain_percent,
                nominal_stress_pa
            )
        )

    output.close()
    odb.close()
else:
    raise RuntimeError('ODB file was not found: ' + odb_path)
