@echo off
call C:\\Users\GeeKandaa\anaconda3\Scripts\activate.bat C:\\Users\GeeKandaa\anaconda3\
call activate tensorflow
cd C:\\Users\GeeKandaa\python_scripts\Image_CNN

@REM FOR /L %%x IN (1,1,1) DO python Image_CNN_Example_Tensorflow.py
setlocal ENABLEDELAYEDEXPANSION
set /a a=1
FOR /L %%x IN (1,1,5) DO (
	set /a b=%%x*!a!
	set num_var=0.000!b!
	python Auto_Parameterise_Image_CNN.py --optimiser lr:!num_var!
)

PAUSE