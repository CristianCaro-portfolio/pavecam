# ADR 001: Pavement sensors as labels, leave one site out as the test

Status: accepted

## Context

Camera road condition models are usually trained on hand labeled frames and tested on a
random split. Hand labels are slow and subjective, and a random split puts frames of the same
camera, minutes apart, in train and test.

## Decision

Use the RWIS pavement sensor at the same site as the label. Map NTCIP surface codes to three
classes (dry, wet, frozen) and drop Other, Error and missing codes. Pair a frame with a label only
if they are within 10 minutes. Evaluate with leave one site out cross validation, and report a
random split only as a leak check.

## Why

* Labels are free, objective and come at 5 minute resolution for every winter archived by IEM.
* The real use case is a camera that has no sensor. Leave one site out is the only split that
  matches it.
* Measured cost of the wrong split: the image probe scores 0.78 macro F1 on a random split and
  0.45 on unseen cameras.

## Consequences

The label is what the sensor measures, not what the camera sees. Frost on bare pavement and a
dry deck in a snowy scene are labeled correctly but look wrong. This lowers image scores and is
documented as a limitation instead of being cleaned away.
