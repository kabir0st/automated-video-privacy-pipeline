"""PyQt6 timeline editor for the union pipeline.

  theme      colours + stylesheet
  frames     random-access frame source with a small cache (scrub/playback)
  worker     QThread owning the models; runs analyse / offline / export /
             propagate jobs and reports through signals
  canvas     frame view with track overlay, selection and box drawing
  timeline   track lanes + coverage strip + playhead
  inspector  selected-track panel (suspicion breakdown, thumbnails, actions)
  window     MainWindow wiring everything together
"""
