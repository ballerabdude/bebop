import traceback

# ============================================================
# Helpers
# ============================================================
def collect_bodies_in_occurrence(occurrence, bodies_list):
    """All solid bodies in this occurrence + child occurrences (as proxies)."""
    for i in range(occurrence.bRepBodies.count):
        b = occurrence.bRepBodies.item(i)
        if b.isSolid and b.isValid:
            bodies_list.append(b)
    for i in range(occurrence.childOccurrences.count):
        collect_bodies_in_occurrence(occurrence.childOccurrences.item(i), bodies_list)


def collect_native_bodies(component, bodies_list):
    """All native (non-proxy) bodies in a component and its sub-components."""
    for i in range(component.bRepBodies.count):
        bodies_list.append(component.bRepBodies.item(i))
    for i in range(component.occurrences.count):
        collect_native_bodies(component.occurrences.item(i).component, bodies_list)


def merge_bodies_in_component(occurrence):
    """
    Merge all bodies inside this occurrence into one BRepBody in its component.
    Returns (success: bool, message: str).
    """
    comp = occurrence.component
    comp_name = occurrence.name

    # Collect all bodies (assembly-context proxies, so geometry includes any
    # transforms from sub-occurrences)
    proxy_bodies = []
    collect_bodies_in_occurrence(occurrence, proxy_bodies)

    if len(proxy_bodies) == 0:
        return True, f'{comp_name}: no bodies (skipped)'
    if len(proxy_bodies) == 1:
        return True, f'{comp_name}: already 1 body'

    n_input = len(proxy_bodies)

    # ---- Step 1: Build merged TBrepBody via TemporaryBRepManager ----
    temp_brep = adsk.fusion.TemporaryBRepManager.get()
    accumulator = temp_brep.copy(proxy_bodies[0])
    union_failures = 0

    for b in proxy_bodies[1:]:
        tool = temp_brep.copy(b)
        try:
            ok = temp_brep.booleanOperation(
                accumulator, tool,
                adsk.fusion.BooleanTypes.UnionBooleanType
            )
            if not ok:
                union_failures += 1
        except Exception:
            union_failures += 1

    # ---- Step 2: Add the merged body to the component (NO BaseFeature) ----
    try:
        new_body = comp.bRepBodies.add(accumulator)
        new_body.name = comp.name + '_merged'
    except Exception as e:
        return False, f'{comp_name}: failed to add merged body — {str(e)}'

    # ---- Step 3: Delete the original bodies ----
    # We must delete using native references (not proxies). Collect native
    # bodies from this component and its sub-components, except the new one.
    natives_to_delete = []
    for i in range(comp.bRepBodies.count):
        b = comp.bRepBodies.item(i)
        if b != new_body and b.isSolid:
            natives_to_delete.append(b)
    for i in range(comp.occurrences.count):
        collect_native_bodies(
            comp.occurrences.item(i).component, natives_to_delete
        )

    deleted = 0
    for b in natives_to_delete:
        try:
            b.deleteMe()
            deleted += 1
        except Exception:
            pass

    note = f' ({union_failures} union(s) failed → kept as lumps)' if union_failures else ''
    return True, (
        f'{comp_name}: merged {n_input} → 1 body{note}, '
        f'deleted {deleted} original(s)'
    )


# ============================================================
# Main
# ============================================================
if design.designType != adsk.fusion.DesignTypes.DirectDesignType:
    ui.messageBox(
        'This script is for direct modeling.\n'
        'Your design is in parametric mode — switch via\n'
        '  Right-click document → "Do not capture Design History"\n'
        'Or use the parametric version of this script.'
    )
else:
    ui.messageBox(
        'Direct Modeling mode confirmed.\n\n'
        'Select the components you want to merge into single bodies, then click OK.\n\n'
        'Each component will end up with exactly ONE body for URDF export.'
    )

    selected = []
    for i in range(ui.activeSelections.count):
        ent = ui.activeSelections.item(i).entity
        if hasattr(ent, 'objectType') and 'Occurrence' in ent.objectType:
            selected.append(ent)

    if not selected:
        ui.messageBox('No components selected. Cancelled.')
    else:
        progress = ui.createProgressDialog()
        progress.cancelButtonText = 'Cancel'
        progress.isCancelButtonShown = True
        progress.show(
            'Merging bodies (direct mode)',
            'Starting...', 0, len(selected), 0
        )
        adsk.doEvents()

        results = []
        for idx, occ in enumerate(selected):
            if progress.wasCancelled:
                break
            progress.progressValue = idx
            progress.message = f'Processing {idx+1}/{len(selected)}: {occ.name}'
            adsk.doEvents()

            try:
                ok, msg = merge_bodies_in_component(occ)
                results.append(('✓' if ok else '✗', msg))
            except Exception as e:
                results.append((
                    '✗',
                    f'{occ.name}: EXCEPTION — {str(e)}\n{traceback.format_exc()}'
                ))

        progress.hide()

        ok_count = sum(1 for r in results if r[0] == '✓')
        summary = '\n'.join(f'{r[0]} {r[1]}' for r in results)
        ui.messageBox(
            f'Done. {ok_count}/{len(results)} components processed successfully.\n\n{summary}'
        )