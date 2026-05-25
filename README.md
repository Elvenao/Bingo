#Bingo

This project is an implementation of 3 computer vision models:

detectCard.pt
detectCircles.pt

These first two were trained with pretained YOLO models with anotations made through label-studio.

cnnModel.pth

This last one was made from scratch using a CNN.

The PipeLine goes like this:
    ________________       ___________________       ________________________
    |Card Detection|  ->   |Circles Detection|   ->  |Number Detection (CNN)|
    ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯       ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯       ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯

The file app.py is the implementation of theses 3 models which has a simple interface that implements the game of Bingo.

This interface implentations has 2 steps. The first step consist in registering the cards through a camera where the identification is made automatically by the 3 models. The proccessInput.py has the task of returning the crops of every single circle going from de card detection model to the circles detection model. After that we predict which number is in each crop by using the cnn.

The requirements needed are in the requirements.txt file. Please use this command:
pip install -r requirements.txt