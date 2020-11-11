### Annotating images (Optional if you want to annotate your own images)

To annotate images, we will be using the labelImg package. For installation and usage instruction, please refer to [LabelImg github repo](https://github.com/tzutalin/labelImg)

Once you have collected all the images to be used to train your model (ideally more than 100 per class), place them inside the folder `images`. It is good practice to name your files with a sequence number for easy processing later on, e.g. raccoon-1.jpg, raccoon-2.jpg, etc. Choose PascalVOC format for your annotations. Start annotating, and you should have an .xml file generated for each of the image you annotate. 

Copy all images, together with their corresponding *.xml files, and place them inside the ``images`` folder and ``annotations`` folder.