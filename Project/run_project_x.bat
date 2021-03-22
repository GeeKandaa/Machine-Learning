@echo off
call C:\\Users\GeeKandaa\anaconda3\Scripts\activate.bat C:\\Users\GeeKandaa\anaconda3\
call activate tensorflow
cd C:\Users\GeeKandaa\Documents\Uni Project\Git Repository\clone\Machine-Learning

@REM FOR /L %%x IN (1,1,1) DO python Image_CNN_Example_Tensorflow.py
setlocal ENABLEDELAYEDEXPANSION
set /a a=1
FOR /L %%x IN (1,1,5) DO (
	set /a b=%%x*!a!
	set num_var=!b!
	python Auto_Parameterise_Image_CNN.py --model class_weights:[!num_var!,1] epoch:10
)
FOR /L %%x IN (1,1,5) DO (
	set /a b=%%x*!a!*2
	set num_var=0.!b!
	python Auto_Parameterise_Image_CNN.py --model class_weights:[!num_var!,1] epoch:10
)
FOR /L %%x IN (1,1,5) DO (
	set /a b=%%x*!a!
	set num_var=!b!
	python Auto_Parameterise_Image_CNN.py --model class_weights:[1,!num_var!] epoch:10
)
FOR /L %%x IN (1,1,5) DO (
	set /a b=%%x*!a!*2
	set num_var=0.!b!
	python Auto_Parameterise_Image_CNN.py --model class_weights:[1,!num_var!] epoch:10
)

PAUSE