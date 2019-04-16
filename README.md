# Commander-AI

This is the version that was submitted as my final project.

Details in the playGame.py file on how to play the game (Not yet updated for the AI abstract class)

Highly recommend using Anaconda Python 3.5 distribution. It should come with (and configure) everything you need to run this project.

Download Anaconda here: https://www.anaconda.com/download/

You will need to downgrade to 3.5 because of Theano. In the Anaconda command prompt (program is called Anaconda Prompt) type "conda install python=3.5" to downgrade to Python 3.5.

To run Python using Anaconda, start the program called Anaconda Navigator (might take a minute to actually launch) and select an IDE. Then open the Python program and run it.

You can create your own AI's to play the game by using the AI class as the parent and defining a get_move method in your class. Then pass the Engine an instance of your class at initialization in the controllerArray argument.

______________________________________________________

Libraries needed:

Theano 1.x

Pandas

numpy (included in Theano)

