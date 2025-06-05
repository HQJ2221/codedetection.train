# ArUco: Fiducial Marker

## Attributes

- easy to print and detect
- binary code with ID
- Usage: detection & pose estimation

> what is pose estimation?

- e.g. take a picture for an ArUco mark using a camera
- 4 points allows, and we can estimate the transformation from world coordinate to camera coordinate




## Design purpose

- Less bits than QR code? can detect as far as possible
- More CRC bits than ID bits? for rotation check
- Occlusion(阻挡) problem?
- Embedding markers? using fractal(碎片化) marker
- Planar surface has limited angle visibility.(太刁钻的角度检测不到) 
    - Actually it's a feature: customize a polyhedron and place marker on each surface, then we can estimate the path and movement(even rotation) of this polyhedron.



## Application

- Camera pose estimation
- Handwriting prediction
- 3D modeling: combine **keypoints**(in scenes) and markers to detect(modeling) specific scenes.
    - So is keypoints really a must in detection?
- Robotic: SLAM(simutaneous localization and mapping)
- Virtual Welding(焊接): 模拟机械臂焊接零件



