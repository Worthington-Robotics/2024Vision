# 2024Vision
The AprilTag vision code for the 2024 robot

This code is very much adapted from the *Northstar* system developed by [Mechanical Advantage](https://github.com/Mechanical-Advantage). Things like multithreading have been added to change the performance characteristics.

The code is designed to be run on a coprocessor like an Orange Pi along with a global-shutter camera like an ArduCam. In order to use features like GStreamer, it is required that you build OpenCV yourself from source.
