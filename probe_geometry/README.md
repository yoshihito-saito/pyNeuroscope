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
