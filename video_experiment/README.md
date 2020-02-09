# Beat Tracking

## Procedure
1. load audio
2. using spectral based onset detection and resample it
3. use PLP to enhance the periodicty
4. use PLP novelty to compute tempogram
5. output timestamp, magnitude, tempo, and beats_label to csv file


### Predominant Local Pulse

The PLP method analyzes the onset strength envelope in the frequency domain to find a locally stable tempo for each frame. These local periodicities are used to synthesize local half-waves, which are combined such that peaks coincide with rhythmically salient frames (e.g. onset events on a musical time grid). The local maxima of the pulse curve can be taken as estimated beat positions.

This method may be preferred over the dynamic programming method of beat_track when either the tempo is expected to vary significantly over time. Additionally, since plp does not require the entire signal to make predictions, it may be preferable when beat-tracking long recordings in a streaming setting.
