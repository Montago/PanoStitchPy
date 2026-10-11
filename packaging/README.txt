PanoStitch {version} - panoramas from DJI drones
https://github.com/Montago/PanoStitchPy
================================================

Stitches the single images from the drone's panorama modes into one image:
  360° sphere (33 images), 180° panorama (21), wide angle (9) and vertical (3).
360°, 180° and wide-angle results carry 360° metadata, so they can be shown in
Facebook, Google Photos, Kuula, VR headsets etc. Developed and tested with the DJI Air 3.


GETTING STARTED
---------------
Double-click PanoStitch.exe. Nothing needs to be installed - the whole folder is the program.
Always move or copy the whole folder (PanoStitch.exe does not work without the _internal folder).
Tip: right-click PanoStitch.exe > "Show more options" > "Send to" > "Desktop (create shortcut)".


HOW TO USE
----------
1. Choose the folder with a panorama - or the whole PANORAMA folder from the drone's SD card
   to find all panoramas. You can also drag a folder onto the window.
   Use the ORIGINAL single images (e.g. PANORAMA\001_0174\PANO_0001.JPG ...).
2. Choose type, quality and exposure and press "Stitch". The type is detected automatically
   and shown next to each panorama in the list.
3. The result is saved as <date>_<folder>_<type>.jpg in the "Save to" folder
   (default: Pictures\PanoStitch) and shown in the built-in 360° viewer.
4. Try the views "Tiny planet" and "Inverted" above the viewer: drag sideways to turn, up/down to tilt,
   scroll to zoom. "Save view..." saves what is inside the frame as a high-resolution image.

Time per 360° panorama (33 images): Fast approx. 3 min, Standard approx. 4 min, High approx. 10 min.


GOOD TO KNOW
------------
- The program creates no working folders and never writes to the image folder / SD card.
  The only things saved are the finished panorama (and your settings in Windows).
- Partial panoramas (180°, wide angle, vertical) are cropped so there are no black borders.
- If images are missing (e.g. the drone stopped early), those areas are left black.
- The drone's camera cannot point straight up; for 360° the gap in the sky is filled with smooth sky.
- "Correct for the drone drifting" makes the ground close to the drone sharper when it drifts in the wind.
- "Exposure": "Keep the drone's exposure on the ground" gives a bright landscape but a larger burnt-out sun;
  "Balanced" keeps more colour in the sky but makes the ground a little darker.
- The first start can take a few seconds while Windows scans the program.


LICENSE
-------
PanoStitch is MIT licensed (LICENSE.txt). It bundles third-party components under their own licenses,
see THIRD_PARTY_NOTICES.txt and the THIRD_PARTY_LICENSES folder.
