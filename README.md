Before run the code, you need to install the required packages.
Keras, gym, and others can be installed by "conda" or "pip".
To install Box2d, follow the steps:

   git clone https://github.com/pybox2d/pybox2d 
   
   cd pybox2d/ 
   
   python setup.py 
   
   clean python setup.py build 
   
   sudo python setup.py install
   
   
Then, just use "python project2_xy3.py" command to run it. I am using python3. 

The code will generate reward.dat, action.dat, epsilon.dat to save the results of each episode. 

The well trained results including epsilons, states, actions are saved as "stats.pickle" file. 

Then, use plot2,3,4,5 to plot different kinds of figures. 

