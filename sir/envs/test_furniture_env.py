import mujoco
import mujoco.viewer
from pathlib import Path
import xml.etree.ElementTree as ET

# Import the Franka Panda model from robot_descriptions
from robot_descriptions import panda_mj_description

# Paths
scene_xml_path = Path(__file__).parent / "furniture_assembly.xml"
franka_xml_path = panda_mj_description.MJCF_PATH
franka_mesh_dir = Path(franka_xml_path).parent / "assets"

print(f"Loading Franka model from: {franka_xml_path}")
print(f"Franka mesh directory: {franka_mesh_dir}")
print(f"Loading scene from: {scene_xml_path}")

# Load both XML files
scene_tree = ET.parse(scene_xml_path)
scene_root = scene_tree.getroot()

# Update compiler meshdir to point to Franka meshes from Menagerie
compiler = scene_root.find('compiler')
if compiler is not None:
    compiler.set('meshdir', str(franka_mesh_dir))

franka_tree = ET.parse(franka_xml_path)
franka_root = franka_tree.getroot()

# Find worldbody in both
scene_worldbody = scene_root.find('worldbody')

# Remove the placeholder robot from scene
placeholder_robot = scene_worldbody.find(".//body[@name='robot_base']")
if placeholder_robot is not None:
    scene_worldbody.remove(placeholder_robot)
    print("Removed placeholder robot")

# Create a body for the Franka positioned on the table
franka_body = ET.Element('body', {'name': 'franka_mount', 'pos': '-0.4 0 0.42'})

# Copy Franka's worldbody content into our positioned body
franka_worldbody = franka_root.find('worldbody')
if franka_worldbody is not None:
    for child in list(franka_worldbody):
        franka_body.append(child)

# Add Franka to scene
scene_worldbody.append(franka_body)

# Copy Franka default classes to scene (must come before assets that reference them)
franka_default = franka_root.find('default')
if franka_default is not None:
    scene_default = scene_root.find('default')
    if scene_default is None:
        # Insert default section early in the document (before asset)
        scene_default = ET.Element('default')
        # Find asset position and insert before it
        asset_idx = list(scene_root).index(scene_root.find('asset'))
        scene_root.insert(asset_idx, scene_default)
    # Copy all default classes from Franka
    for default_elem in list(franka_default):
        scene_default.append(default_elem)

# Copy Franka assets to scene
franka_assets = franka_root.find('asset')
scene_assets = scene_root.find('asset')
if franka_assets is not None and scene_assets is not None:
    for asset in list(franka_assets):
        scene_assets.append(asset)

# Copy Franka actuators if they exist
franka_actuators = franka_root.find('actuator')
if franka_actuators is not None:
    scene_actuators = scene_root.find('actuator')
    if scene_actuators is None:
        scene_actuators = ET.SubElement(scene_root, 'actuator')
    for actuator in list(franka_actuators):
        scene_actuators.append(actuator)

# Copy other important sections (contact, equality, etc.)
for section_name in ['contact', 'equality', 'tendon', 'sensor']:
    franka_section = franka_root.find(section_name)
    if franka_section is not None:
        scene_section = scene_root.find(section_name)
        if scene_section is None:
            scene_section = ET.SubElement(scene_root, section_name)
        for elem in list(franka_section):
            scene_section.append(elem)

# Save combined model
combined_xml_path = Path(__file__).parent / "furniture_assembly_with_franka.xml"
scene_tree.write(combined_xml_path)
print(f"Saved combined model to: {combined_xml_path}")

# Load the combined model
print("\nLoading combined model...")
model = mujoco.MjModel.from_xml_path(str(combined_xml_path))
data = mujoco.MjData(model)

# Launch viewer
print(f"✓ Model loaded: {model.nq} DOFs, {model.nbody} bodies")
print(f"✓ Actuators: {model.nu}")
print("\nScene loaded successfully! Press Ctrl+C to close.")
print("Use the viewer to interact with the scene.")
mujoco.viewer.launch(model, data)
