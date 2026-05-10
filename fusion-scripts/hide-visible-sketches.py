# Hide all visible sketches within a user-selected component
# Prompts user to select an occurrence/component, then traverses it recursively

hidden_count = 0
checked_count = 0

def hide_sketches_in_component(component):
    global hidden_count, checked_count
    
    for sketch in component.sketches:
        checked_count += 1
        if sketch.isVisible:
            try:
                sketch.isVisible = False
                hidden_count += 1
                print(f"Hidden sketch: '{sketch.name}' in component '{component.name}'")
            except Exception as e:
                print(f"Could not hide sketch '{sketch.name}': {str(e)}")

# Prompt the user to select an occurrence (component instance) in the browser/canvas
selection = ui.selectEntity(
    'Select a component to hide sketches within (click a component in the browser or canvas)',
    'Occurrences'
)

if selection:
    entity = selection.entity
    
    # Resolve the selection to a Component
    target_component = None
    if hasattr(entity, 'component'):
        # It's an Occurrence
        target_component = entity.component
        print(f"Selected occurrence: '{entity.name}' -> component '{target_component.name}'")
    elif hasattr(entity, 'sketches'):
        # It's already a Component
        target_component = entity
        print(f"Selected component: '{target_component.name}'")
    
    if target_component:
        # Hide sketches in the selected component itself
        hide_sketches_in_component(target_component)
        
        # Recursively hide sketches in all sub-occurrences of this component
        for sub_occ in target_component.allOccurrences:
            hide_sketches_in_component(sub_occ.component)
        
        ui.messageBox(
            f'Done!\n\n'
            f'Target component: {target_component.name}\n'
            f'Checked: {checked_count} sketches\n'
            f'Hidden: {hidden_count} visible sketches'
        )
    else:
        ui.messageBox('Could not resolve selection to a component.')
else:
    ui.messageBox('No component selected — operation cancelled.')