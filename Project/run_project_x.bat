@echo off
call C:\\Users\GeeKandaa\anaconda3\Scripts\activate.bat C:\\Users\GeeKandaa\anaconda3\
call activate tenstest
cd C:\Users\GeeKandaa\Documents\Uni Project\Git Repository\clone\Machine-Learning

@REM FOR /L %%x IN (1,1,1) DO python Image_CNN_Example_Tensorflow.py
setlocal ENABLEDELAYEDEXPANSION
set /a a=1
@REM FOR /L %%x IN (1,1,5) DO (
@REM 	set /a b=%%x*!a!
@REM 	set num_var=!b!
@REM 	python AP_3IMAGE_CNN.py --model class_weights:[!num_var!,1,1] epoch:10
@REM )
@REM ren "C:\Users\GeeKandaa\Documents\Uni Project\Git Repository\clone\Machine-Learning\Support_Files\output_data.json" "1_1_5_output_data.json"
@REM FOR /L %%x IN (1,1,5) DO (
@REM 	set /a b=%%x*!a!*2
@REM 	set num_var=0.!b!
@REM 	python AP_3IMAGE_CNN.py --model class_weights:[!num_var!,1,1] epoch:10
@REM )
@REM ren "C:\Users\GeeKandaa\Documents\Uni Project\Git Repository\clone\Machine-Learning\Support_Files\output_data.json" "1_1_10_output_data.json"
@REM FOR /L %%x IN (1,1,5) DO (
@REM 	set /a b=%%x*!a!
@REM 	set num_var=!b!
@REM 	python AP_3IMAGE_CNN.py --model class_weights:[2,!num_var!,1] epoch:10
@REM )
@REM ren "C:\Users\GeeKandaa\Documents\Uni Project\Git Repository\clone\Machine-Learning\Support_Files\output_data.json" "2_1_5_output_data2.json"
@REM FOR /L %%x IN (1,1,5) DO (
@REM 	set /a b=%%x*!a!*2
@REM 	set num_var=0.!b!
@REM 	python AP_3IMAGE_CNN.py --model class_weights:[2,!num_var!,1] epoch:10
@REM )
@REM ren "C:\Users\GeeKandaa\Documents\Uni Project\Git Repository\clone\Machine-Learning\Support_Files\output_data.json" "2_1_10_output_data2.json"
FOR /L %%x IN (1,1,5) DO (
	set /a b=%%x*!a!
	set num_var=!b!
	python AP_3IMAGE_CNN.py --model class_weights:[2,0.8,!num_var!] epoch:10
)
ren "C:\Users\GeeKandaa\Documents\Uni Project\Git Repository\clone\Machine-Learning\Support_Files\output_data.json" "3_1_5_output_data2.json"
FOR /L %%x IN (1,1,5) DO (
	set /a b=%%x*!a!*2
	set num_var=0.!b!
	python AP_3IMAGE_CNN.py --model class_weights:[2,0.8,!num_var!] epoch:10
)
ren "C:\Users\GeeKandaa\Documents\Uni Project\Git Repository\clone\Machine-Learning\Support_Files\output_data.json" "3_1_10_output_data2.json"

PAUSE