# PatchSim
Code for simulating the metapopulation SEIR model. Sample network for US national simulation included. 

A preliminary description of this model is provided in 
https://ieeexplore.ieee.org/document/8031141/

Cite as: Venkatramanan S, Chen J, Fadikar A, Gupta S, Higdon D, Lewis B, Marathe M, Mortveit H, Vullikanti A. Optimizing spatial allocation of seasonal influenza vaccine under temporal constraints. PLoS computational biology. 2019 Sep 16;15(9):e1007111.




## Documentation

The description of the model and a software manual can be found inside the doc/ folder. 

## Dependencies

PatchSim is compatible with both Python 2 and 3. It requires numpy and pandas. 


## Testing

Please use the different test*.py scripts to test the different functionalities of PatchSim. For more details on these features please check the software manual.


## US National simulation

To test the US national simulation, extract the zip file test/US_county.zip and run "python test_det_US.py" from inside test/ folder. 
