# Probe Geometry

Put probe site geometry JSON files in this folder. The filename stem is used as
the probe type in pyNeuroscope, for example `poly2.json` appears as `poly2`.

## Slot-based geometry

Use `slot` for geometry that is repeated for every channel group/shank. Channels
are mapped by their order inside each group.

```json
{
  "name": "poly2",
  "units": "um",
  "group_pitch_um": 250,
  "sites": [
    { "slot": 0, "x": -10, "y": 0 },
    { "slot": 1, "x": 10, "y": 20 }
  ]
}
```

## Explicit channel geometry

Use `channel` when the file already contains probe-local channel coordinates.

```json
{
  "name": "my_probe",
  "units": "um",
  "sites": [
    { "channel": 0, "x": 0, "y": 0 },
    { "channel": 1, "x": 20, "y": 20 }
  ]
}
```

Coordinates are probe-local. pyNeuroscope offsets channels automatically when
multiple probes are configured.

## Built-in patterns

These probe types are available even without JSON files:

- `poly2`
- `poly3`
- `poly5`
- `staggered`
- `neuropixel`
- `double_sided`
- `neurogrid`

The procedural patterns map each channel group/shank using the channel order in
the XML or channel group editor.

## Session ChannelMap

Use the `Load Session ChannelMap` button in the Recording tab to load an
existing `chanMap.mat`. This is separate from Probe type. pyNeuroscope reads
`chanMap0ind`, `xcoords`, and `ycoords` directly.

## Full Neuropixels geometry and selected channels

Three complete physical layouts are bundled in
`src/pyneuroscope/resources/neuropixels_geometries.json`: Neuropixels 1.0
(960 sites), Neuropixels 2.0 single shank (1280 sites), and Neuropixels 2.0
four shank (5120 sites). Their parameters and site-ID convention follow
[Atlaxis](https://github.com/yoshihito-saito/Atlaxis/tree/main/probes/Neuropixels).
Coordinates are micrometers with +y toward the base; NP2's 200 µm first-site
offset is provisional. The legacy `neuropixel` pattern remains a schematic
based on group order and is separate from these complete layouts.

Select physical site IDs in DAT channel order using `Active site IDs`, or load
an explicit recording map. The geometry catalog assigns no DAT channels by
itself. Atlaxis nested `chanCoords` MAT structures are supported; their
one-based channel field is converted to zero-based DAT IDs without removing
disconnected rows. Site IDs such as `s0_front_0384` are preserved when supplied.

Explicit JSON recording maps use zero-based DAT channels:

```json
{
  "sites": [
    {"channel": 0, "site_id": "s0_front_0384", "x": -24, "y": 4040, "active": true},
    {"channel": 1, "site_id": "s0_front_0385", "x": 8, "y": 4040, "active": true}
  ]
}
```

For a session map, channel IDs are global recording columns. For an individual
probe, channel IDs are local to that probe's DAT block. Do not use the order of
XML anatomical groups as a recording-column map. Atlas-space coordinates are
not supported as probe-local geometry.
