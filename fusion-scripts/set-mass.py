# =============================================================
# Set Component Mass via Custom Material
# =============================================================

# --- 1. Ask the user to select a component (occurrence) ---
sel = ui.selectEntity(
    'Select a component (click on a body or component in the browser/canvas)',
    'Occurrences,Bodies'
)

if sel:
    entity = sel.entity
    targetComp = None
    targetOcc = None

    if entity.objectType == adsk.fusion.Occurrence.classType():
        targetOcc = entity
        targetComp = entity.component
    elif entity.objectType == adsk.fusion.BRepBody.classType():
        targetComp = entity.parentComponent
        for occ in rootComp.allOccurrences:
            if occ.component == targetComp:
                targetOcc = occ
                break
    else:
        ui.messageBox('Please select a component or a body.')
        targetComp = None

    if targetComp:
        compName = targetComp.name
        bodies = [b for b in targetComp.bRepBodies]

        if len(bodies) == 0:
            ui.messageBox('Component "{}" contains no bodies.'.format(compName))
        else:
            # Volume in cm^3 (Fusion internal length unit is cm)
            total_volume_cm3 = sum(b.volume for b in bodies)

            (mass_input, isCancelled) = ui.inputBox(
                'Enter target mass for "{}" in GRAMS:\n\nTotal volume: {:.4f} cm^3'.format(
                    compName, total_volume_cm3),
                'Set Component Mass',
                '100.0'
            )

            if not isCancelled:
                target_mass_g = float(mass_input)

                # density in g/cm^3, then convert to kg/m^3 (Fusion internal density unit)
                density_g_cm3 = target_mass_g / total_volume_cm3
                density_kg_m3 = density_g_cm3 * 1000.0   # <-- FIXED

                materialName = 'MASS_' + compName
                designMaterials = design.materials

                customMaterial = None
                for m in designMaterials:
                    if m.name == materialName:
                        customMaterial = m
                        break

                if customMaterial is None:
                    libs = app.materialLibraries
                    fusionLib = libs.itemByName('Fusion Material Library')
                    if fusionLib is None:
                        fusionLib = libs.item(0)

                    baseMaterial = None
                    for m in fusionLib.materials:
                        if m.name.lower() == 'steel':
                            baseMaterial = m
                            break
                    if baseMaterial is None:
                        baseMaterial = fusionLib.materials.item(0)

                    customMaterial = designMaterials.addByCopy(baseMaterial, materialName)

                # Set density (internal unit: kg/m^3)
                densityProp = customMaterial.materialProperties.itemById('structural_Density')
                if densityProp:
                    densityProp.value = density_kg_m3   # <-- FIXED
                else:
                    ui.messageBox('Could not find density property on material.')

                # Apply material to all bodies
                for b in bodies:
                    b.material = customMaterial

                # Verify
                physProps = targetComp.getPhysicalProperties(
                    adsk.fusion.CalculationAccuracy.HighCalculationAccuracy
                )
                actual_mass_g = physProps.mass * 1000.0

                ui.messageBox(
                    'Mass set successfully!\n\n'
                    'Component: {}\n'
                    'Material:  {}\n'
                    'Volume:    {:.4f} cm³\n'
                    'Density:   {:.6f} g/cm³  ({:.2f} kg/m³)\n'
                    'Target:    {:.4f} g\n'
                    'Actual:    {:.4f} g'.format(
                        compName, materialName, total_volume_cm3,
                        density_g_cm3, density_kg_m3,
                        target_mass_g, actual_mass_g
                    )
                )