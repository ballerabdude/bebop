# Auto-rename the single body in each component to match the component name
# Run this AFTER combining bodies so each component has exactly 1 body

renamed_count = 0
skipped_multi = []
skipped_empty = []

def rename_bodies_in_component(component):
    global renamed_count
    
    # Get solid bodies in this component
    solid_bodies = []
    for i in range(component.bRepBodies.count):
        body = component.bRepBodies.item(i)
        if body.isSolid:
            solid_bodies.append(body)
    
    comp_name = component.name
    
    if len(solid_bodies) == 1:
        # Perfect case - rename the single body to match component
        old_name = solid_bodies[0].name
        if old_name != comp_name:
            solid_bodies[0].name = comp_name
            renamed_count += 1
    elif len(solid_bodies) > 1:
        # Multiple bodies - append index
        skipped_multi.append(comp_name)
        for idx, body in enumerate(solid_bodies):
            body.name = f'{comp_name}_{idx+1}'
            renamed_count += 1
    else:
        skipped_empty.append(comp_name)

# Process root component first
rename_bodies_in_component(rootComp)

# Then process all unique components (avoid duplicate renames for instanced components)
processed_components = set()
for i in range(rootComp.allOccurrences.count):
    occ = rootComp.allOccurrences.item(i)
    comp = occ.component
    if comp.id not in processed_components:
        processed_components.add(comp.id)
        rename_bodies_in_component(comp)

# Report results
msg = f'✅ Renamed {renamed_count} bodies\n\n'
if skipped_multi:
    msg += f'⚠️  Components with multiple bodies (named with suffix):\n'
    msg += '\n'.join(f'  • {n}' for n in skipped_multi) + '\n\n'
if skipped_empty:
    msg += f'ℹ️  Components with no solid bodies (skipped):\n'
    msg += '\n'.join(f'  • {n}' for n in skipped_empty)

ui.messageBox(msg)