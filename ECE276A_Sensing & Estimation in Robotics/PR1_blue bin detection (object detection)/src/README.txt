For pixel_classification:

All model related functions are written in "model_setup.py".

To train the model (update model_parameters_b and model_parameters_w), only need to run "pixel_classification/Run_This_to_Train.py".

*For the first time, both 'model_parameters_b.txt' and 'model_parameters_w.txt' are not created yet, please uncommon first section in "Run_This_to_Train.py" to create 'model_parameters_b.txt' and 'model_parameters_w.txt' in local directory.

Then you can use 'classify' function from 'pixel_classifier.py' to identify the color of pixel.

For bin_detection:

I used same "model_setup.py" to create model, however, this time I add extra 5 colors (totally we have Red = 1, Green = 2, Blue = 3, Skyblue = 4, Black = 5, White = 6, Yellow = 7, Gray = 8).

*This time color dataset will be search under "training_color" directory, please keep it this form.
-bin_detection
 --training_color
   --red
   --green
   --blue
   --skyblue
   --black
   --white
   --yellow
   --gray

Same as pixel_classification, to train the model (update model_parameters_b and model_parameters_w), only need to run "bin_detection/Run_This_to_Train.py".

Then you can call 'get_bounding_boxes' function from 'bin_detector.py' to get blue-bin region coordinates.