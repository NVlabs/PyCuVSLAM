API Reference
=============

.. module:: cuvslam

This module provides Python bindings for the cuVSLAM library.

Data Structures
---------------

.. autoclass:: Pose
   :members:

.. autoclass:: Distortion
   :members:

.. autoclass:: Camera
   :members:

.. autoclass:: ImuCalibration
   :members:

.. autoclass:: Rig
   :members:

.. autoclass:: PoseEstimate
   :members:

.. autoclass:: ImuMeasurement
   :members:

.. autoclass:: Landmark
   :members:

.. autoclass:: Observation
   :members:

.. autoclass:: cuvslam.core.Odometry.MulticameraMode
   :members:

.. autoclass:: cuvslam.core.Odometry.OdometryMode
   :members:

.. autoclass:: cuvslam.core.Odometry.Config
   :members:

.. autoclass:: cuvslam.core.Odometry.RGBDSettings
   :members:

.. autoclass:: cuvslam.core.Slam.Config
   :members:

.. autoclass:: cuvslam.core.Slam.Metrics
   :members:

.. autoclass:: cuvslam.core.Slam.LocalizationSettings
   :members:

Tracker class
-------------

.. autoclass:: Tracker
   :members:
   :undoc-members:
   :exclude-members: merge_maps, save_map, localize_in_map, set_slam_pose, OdometryMode, MulticameraMode

Functions
---------

.. autofunction:: get_version

.. autofunction:: set_verbosity

.. autofunction:: warm_up_gpu
