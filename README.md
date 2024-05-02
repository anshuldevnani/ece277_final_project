README

Prereqs - CUDA 12.3 (and added to PATH), Python 3.0 or above (and added to path), Visual Studio 2019 or greater, Cmake

Note: The project has been test on ITS-E4309-02. Cuda 12.3 exists on PATH but you might need to download and add python

Steps to RUN

1. Grab the submission zip folder
2. Open the CMakeLists.txt in CMake GUI and fill in the paths to the source and build binaries
3. Click configure and generate
4. Open up the FloydWarshallCuda.sln in Visual Studio 2019 
5. Right click on the FloydWarshallCuda solution and click properties
6. Click Advanced and change the Target File Extension to .pyd
	- Make sure the configuration is set to Release
7. Build in Release
8. Make sure the FloydWarshallCuda.pyd exists in build/Release folder
9. To instrument the code, open up a cmd prompt in the top level dir, and run the following

python profile.py --vertices-list "16, 32, 64, 128, 256, 512, 1024"

NOTE: vertices above 1024 will lead to long CPU FW algo runtimes