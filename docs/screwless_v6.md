# v6: screwless finger mounting and a fit-on jaw tag cap

Status: CAD, fit- and collision-checked, **not printed yet**. Build: `cd cad && ../.venv-cad/bin/python build_v6.py`.

![exploded](../cad/renders/v6_exploded.png)

Adapted from the drop-in fit of the PLA claw and mount Akshat sent: his claw's base drops into a matching pocket from
above. Our holder clamps the finger between a top and a bottom plate, so the idea becomes a finger that drops onto
pins and a removable cap that closes the holder over it.

## Parts (per side, left and right)
- **Finger holder**: the Stanford holder with its top plate cut off over the finger (from y 76 forward), the bolt
  holes and nut pockets filled, three 3.9 mm pins standing up from the bottom plate at the finger's own three 4 mm
  holes, and the v5 dovetail rail underneath. The ring mounts (standard and small-hand) fit unchanged.
- **TPU soft finger**: the stock UMI finger (`cad/out_v3/soft_finger_*.stl`), unchanged. It drops onto the pins from
  above; the pins hold it forward/back and sideways, the cap holds it down.
- **Tag cap**: the cut-off piece of top plate, carrying the jaw ArUco tag. Its tongue goes into a slot in the holder's
  top plate, the cap swings down, and a snap arm on its outer edge clicks under the bottom plate. A 0.5 mm pad under
  its back squeezes the finger's solid back strip by 0.1 mm so nothing rattles; the fin ray stays free.
  - The tag sits in a 20.4 mm, 0.2 mm deep recess; the cap's top is 0.2 mm higher than the holder, so a 0.2 mm
    sticker's face lands where `configs/gripper_markers.yaml` already puts tags 6 and 7. No config change.
  - Release: push the snap arm's tip outward from below, swing the cap up, pull it forward.

Finger tip position, jaw travel and the outer envelope are as in v3. The finger now rests on the bottom plate, 0.4 mm
lower than in the v3 CAD (it floated in a 26.2 mm gap); the TCP estimate in the config moves by that much.

## Checks (in CAD)
- Holders and caps are single valid solids; left and right are built from each finger's actual hole positions (the
  right finger sits 0.57 mm off the mirror position in the Stanford assembly).
- Seated: no overlap between cap, holder, pins and finger except the intended 0.1 mm pad squeeze.
- The cap collides with the holder if moved down, back, sideways or up; the finger collides if moved in any
  horizontal direction.
- No collisions over the jaw travel; all eight ring mounts seat, lock and travel clear on the new holders.

## Print
`tmp_print/UMI_v6_holders_caps_PLA.3mf` (git-ignored): 2 holders + 2 caps, PLA+. The holders print on their side, so
the pins lie horizontal (layers along the pin, strongest in bending); the stock picker would have printed them
top-down with the pins hanging in mid-air. STEP / STL in `cad/out_v6/`.

## Not verified
- The fits: pin 3.9 mm in a 4.0 mm TPU hole, 0.15 mm clearance on the tongue, 1.3 mm snap lip, 0.1 mm pad squeeze.
  Print one holder and one cap first.
- Pin strength: three printed 3.9 mm pins carry the grip and pull-out loads that bolts plus clamping friction carried
  before. If one breaks, a 4 mm steel rod in a drilled pin position is the fallback.
- The 3MF has not been opened in Bambu Studio. `sim/assets/meshes` still holds the v3 holders.
