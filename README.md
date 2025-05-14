# pyDomeCraft
A small python script to design geodesic domes in Minecraft.

---
## Background and development
Geodesic domes have been a recent interest of mine, and until I convince my wife that I should build one in our garden, I figured I would make one in Minecraft instead. However; while there are plenty of calculators online to give exact lengths of struts and angles to cut if building in the real world, it is not easily translated to the cubic blocky world of Minecraft.

I initially looked into learning the math behind these calculators, to make a script myself, but then realised I could probably take advantage of CoPilot to help me. The current version of the script, as of May 14th 2025 is something like 99% CoPilot. No need to learn the math, and it even, over a few iterations and prompts, coded all the output too. This was exceptionally quick and easy, but on the other hand could probably be optimized greatly. To begin with, Python is obviously not the fastest language, and pyvista, which I randomly came across while researching for the project, but before engaging CoPilot, may not be the best solution. Still, it works pretty well with a few "holes" in the panels, that are easily filled. For domes up to 70-80 blocks in diameter the time to run the script is ok, although at 80 blocks it does take a couple of minutes to render the 3D model of the blocks.

The script does not fully make the bottom part, where triangles would be cut in half, but it should be easy to fix, and allows for designing openings/doors etc. I figured this was manageable to finish in person if needed, by anyone willing to take on a project such as a dome like this.

---

How to use:
The script is selfcontained in the single .py file, however relies on numpy, pyvista and matplotlib to be installed (with their dependencies). Recommend doing it in a venv, but I didn't bother myself and just ran it straight. I have run it with both python 3.12 and 3.13 with no issues, although there is a warning about one of the calls in numpy being deprecated. I may fix this in the future.

Run the .py file in a shell and it will ask for input about the dome diameter (in blocks) and the frequency (2v, 3v, 4v tested so far) ([More info](https://www.domerama.com/geodesic-dome-size-and-frequency/)).

Fairly quickly it should calculate the amount of edges and triangular panels, and give an image showing the dome in a wireframe design like this. The 3D rendering by pyvista allows rotation along all angles to see the dome from the side, the top, or the bottom.

![image](https://github.com/user-attachments/assets/2bb2c3d1-35af-4170-95a8-e93c9f2f32ca)

When closing this window it will output the number of blocks needed and start working on showing the dome in block form. This can take a while for larger domes or domes with higher frequency. Eventually it should pop up with a new window showing the dome in block form, also in 3D rendering.

![image](https://github.com/user-attachments/assets/81ad4951-aa83-4a0e-af9d-323b1736b5d8)
![image](https://github.com/user-attachments/assets/7b6db465-8f2b-45c3-a7d6-075f1c8f7ca7)

Finally, when closing the block rendition, it should will take a few moments more, and then show a 2D representation from above, with a slider at the bottom to move through each layer, effectively giving a layer by layer build guide. Each layer can be saved as an image if desired.

![image](https://github.com/user-attachments/assets/710a2aab-1e6f-456f-89c8-df803bdf3060)
![image](https://github.com/user-attachments/assets/2196891d-fe2f-48e8-85ab-3f72e258a5b7)

And that is it. Hope it is useful to someone, and I very much welcome any contributions, refactoring, rewriting in another (preferably faster) language etc. Realistically I do not have the time, nor the need to do much of it myself, since this seems to work perfectly well for my usecase.










