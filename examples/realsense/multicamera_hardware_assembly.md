# Multi-Camera RealSense Hardware Assembly Guide for Jetson Orin

<img src="../../assets/multicam_battery_lcd.png" width="400" alt="Complete Setup">

This tutorial demonstrates how to build a simple, low-cost, hardware-synchronized multicamera system using three Intel RealSense D435/D455 cameras and a Jetson Orin AGX/Nano Developer Kit. We encourage users to adapt this guide to other camera configurations and hardware setups and share these projects with the community

> **Note**: this tutorial demonstrates all steps on a Jetson Orin Nano Development Board, a similar setup is also available for the Jetson AGX Orin Development Kit. Refer to the [3D-printed frame](../../assets/3rs_435-55_Orin_AGX.stl) and the step-by-step [AGX assembly schematic](../../assets/assembly_guide_agx.png)

This configuration sets up one camera as the primary device, providing synchronization signals to the dependent cameras. The setup uses standard jumper wires, connectors, and a breadboard for easy, solder-free assembly. For more details and additional synchronization options, please refer to the [official RealSense multicamera documentation](https://dev.intelrealsense.com/docs/multiple-depth-cameras-configuration)

---

## 1. Materials Needed

### Base Station
- **Jetson Orin Nano Developer Kit** (1x)
- **3D-printed mounting [frame](../../assets/3rs_435-55_Orin_Nano.stl)** (1x)
- **M2.5x12 screws with nuts** (4x)

### Cameras and Connectors
- **Intel RealSense D435 or D455 cameras** (3x)
- **JST SHR-09V-S connectors** ([link](https://www.digikey.com/en/products/detail/jst-sales-america-inc/SHR-09V-S/759888)) (3x)
- **AWG jumper wires (socket-to-socket, 12")** ([link](https://www.digikey.com/en/products/detail/jst-sales-america-inc/ASSHSSH28K305/6009453)) (6x)
- **M3x6 screws** (6x)

### Sync Module (Required for more than 2 cameras)
- **Breadboard mini** (1x)
- **Jumper wire kit** (1x)
- **20kΩ resistors** (3x)
- **22nF capacitors** (3x)

### Autonomous Power Supply
- **DC-DC Buck-Boost Converter (7.4V → 19V, ≥25W)** (1x)
- **7.4V LiPo battery** (1x)
- **5.5mm x 2.1mm Male DC power plug** (1x)

### Tools
- Screwdriver, wire cutters, tweezers

---

## 2. Preliminary Preparation

Follow the steps below carefully before assembling your setup:

- **Remove the plastic frame from the Jetson Nano Developer Kit**

  *Tip: Remember to detach the WiFi antenna before removal. Reattach it to the M2 modem after completing the assembly in Step 5*  
  <img src="../../assets/jetson_no_frame.png" width="400" alt="Jetson without frame">

- **Connect AWG wires to JST connectors**:  
  Gently attach AWG wires to pins 5 and 9 of JST connectors  
  *Tip: Connect JST connectors to RealSense cameras after completing assembly in Step 5*   
  <img src="../../assets/rs_connector.png" width="800" alt="RealSense Connector">

- **Assemble synchronization module on BreadBoard Mini**:
  >**Note**: If using only 2 cameras, you can connect them directly (see the right-side image below), ensure the corresponding pins match

  Follow the provided circuit diagram for correct assembly of the synchronization module: 
 
  <img src="../../assets/sync_module.png" width="400" alt="Synchronization Module Diagram">
  <img src="../../assets/sync_module_real.png" width="800" alt="RealSense Connector Module">

## 3. Assembly Instructions

Follow the step-by-step assembly schematic below to connect all prepared components:

<img src="../../assets/assembly_guide_nano.png" width="700" alt="Nano Assembly Schematic">

*Tip (Jetson Nano): when placing the breadboard mini on the 3D-printed frame during Step 4, position it closer to the GPIO rather than centered. This ensures easy access to the Reset pin for future JetPack reinstallation.* 

## 4. Completed Setup

Your final assembled setup should look like this and is ready for real-time execution of [PyCuVSLAM multicamera odometry](./README.md#running-multicamera-odometry)

<img src="../../assets/jetson_with_cameras.png" width="600" alt="Jetson Setup with Cameras">

## Additional Notes

- **Camera compatibility:** both available 3D printed frames (for [Nano](../../assets/3rs_435-55_Orin_Nano.stl) and [AGX](../../assets/3rs_435-55_Orin_AGX.stl)) support mounting RealSense D435 or D455 cameras - choose whichever option suits your needs
- **Two-camera configuration:** when using only 2 cameras, the synchronization module is optional. Direct connection between camera connectors is possible
- **USB camera troubleshooting:** occasionally, USB cameras may fail to start. If this occurs, briefly disconnect and reconnect them from the USB port
- **Autonomous power:** the Jetson Orin Nano Development Kit only supports power supply through the DC Jack connector. Use the autonomous power supply components listed in Section 1 for autonomous operation