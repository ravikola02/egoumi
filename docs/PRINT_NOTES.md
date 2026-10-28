# UMI pinch v3: printing on Bambu Lab H2D (0.4 mm nozzle)

## Files
| File | Filament | Parts |
|---|---|---|
| UMI_print1_covers_PLA.3mf | PLA+ | bottom_plate, top_cover (mirror flaps removed) |
| UMI_print2_internals_PLA.3mf | PLA+ | grip, slider, cube, finger_holder_left_thumb, finger_holder_right_index, gear L/R, linkage L/R |
| UMI_print3_fingers_TPU.3mf | TPU 95A | soft_finger_left, soft_finger_right |

The parts are already oriented for printing and placed inside the plate area both H2D nozzles can reach.
The GoPro dummy (camera_dummy) is not included because it is not printed.

## In Bambu Studio
1. Select presets: printer **Bambu Lab H2D 0.4 nozzle**, process **0.20mm Standard @BBL H2D**,
   filament **eSUN PLA+ @BBL H2D** for prints 1–2 and **Generic TPU @BBL H2D** for print 3
   (external spool, not the AMS).
2. Open the .3mf. Bambu Studio shows "not from Bambu Lab, load geometry data only". This is expected:
   the file carries no printer config, so your stock H2D printer preset is used unchanged.
   The per-part settings below are still loaded; parts with overrides show a settings icon in the object list.
3. TPU: set the fan to 0 % (Stanford) in the filament's cooling settings. Use your spool's temperature:
   Stanford used 255 °C for Polymaker TPU 95A, and Bambu's Generic TPU preset is 240 °C (range 200–250 °C).
4. PLA+: Stanford's 190 °C was for a 0.6 mm nozzle on their printer. Bambu's eSUN PLA+ H2D preset uses 220 °C
   at H2D speeds. Keeping the preset temperature is a judgement call, not part of Stanford's spec.

## Per-part settings (Stanford spec translated from 0.6 mm to 0.4 mm nozzle)
| Stanford (0.6 mm nozzle) | In these files (0.4 mm nozzle) |
|---|---|
| 0.3 mm layer | 0.20 mm layer |
| 1.2 mm wall | 3 walls (~1.3 mm) |
| 1.2 mm top / bottom | 6 top + 6 bottom layers, 1.2 mm |
| 20 % gyroid infill | 20 % gyroid |
| grip: 100 % infill, 1.8 mm wall | 100 % infill, 4 walls (~1.8 mm), plus tree supports (see note) |
| finger holders: supports | tree(auto) supports |
| linkages: 2.4 mm wall, 1.8 mm top/bottom | 6 walls (~2.7 mm), 9 top + 9 bottom layers |
| TPU finger: 100 % infill, lines | 100 % infill, "line" pattern |

## Deviations / caveats
- Grip: Stanford lists no supports. In the orientation used here (foot down) about 30 cm² under the top
  flange overhangs more than 45°, so tree supports are on. Turn them off if you print it Stanford's way.
- Finger holders print upside down (flat top face on the bed). Supports fill the soft-finger slot,
  the ring bores and the thumb arm.
- bottom_plate: Stanford's original CAD has 10 non-manifold mesh edges (two pocket walls meet along a
  zero-thickness line). Bambu Studio may report repaired errors; it still slices.
- Mirrors removed: Stanford's UMI pipeline uses the side mirrors (implicit stereo; mirror regions are
  processed in the observations). Disable or adjust that mirror processing if you use their training code.
- These files were not opened in Bambu Studio itself; the app can't run here. They were checked against
  Bambu Studio's 3MF loader source and parsed with lib3mf.
