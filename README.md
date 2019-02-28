# CS231n-solutions-2018
This is a CS231n 2018 assignments solutions.

I would like to share my codes with everyone who is trying to come into the area of Deep Learning.

## Table of Contents
```
1. Differentiation of this repository
2. Errors and Solutions
```

### 1. Differentiation
1) Almost line by line comments
2) Solutions for errors that students might face while doing assignments 
3) Easy code, only using well known functions 
4) Shape of matrices appended after each operation

### 2. Errors and Solutions
**Assignment 1**
```
Error 1) Error regarding the __future__ module
 => solution : 
    -  Move the importing code of __future__ into the first line before all other libraries

Error 2) Cannot import name ‘imread’
 => solution : 
    -  Install Pillow in the terminal 
    -  Edit the “data_utils.py” at line 6
       from scipy.misc import imread   => from scipy.misc.pilutil import imread

Error 3) Indentation
 => solution : 
    -  Install autopep8 from condo
    -  Type this in the terminal 
        autopep8 path/to/file.py —select=E1 —in-place  (for changing format for an individual file)
        or
        autopep8 target_directory —recursive —select=E1 —in-place (to change all format in the given directory)

Error 4) Softmax zero division handling
=> solution :
    -  Add a “self.eta” variable and set it to be over 1
    -  When dividing the exponential of Yi by the sum of exponentials, there comes a zero division error. 
    What you should do here is to add self.eta to all the scores you got by dot product of X and W(weight matrix), 
    and then apply the exponential to the score matrix.
```

**Assignment 2**
```
Error 1) Error regarding the past.builtins in line 3 of “gradient_check.py”
=> solution : 
   -  import “future” module with 
       “ conda install future “
```

**Assignment 3**
```
Error 1) Permission denial when executing ./get_assignment3_data.sh 
=> solution :
  -  chmod +x ./get_assignmnt3_data.sh
  -  chmod +x ./get_coco_captioning.sh
  -  chmod +x ./get_imagenet_val.sh
  -  chmod +x ./get_squeezenet_tf.sh
