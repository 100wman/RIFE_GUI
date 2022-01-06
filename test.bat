@echo off
setlocal EnableDelayedExpansion
set /p input=input_folder: 
set /p output=output_folder: 
set num=1
for /f "tokens=* " %%i in ('dir /b /x /s %input%\*.png') do (
    if num==1 goto break else (
        set input1=%%i
		rem assign your script
        echo python inference.py -i0 !input0! -i1 !input1! -o %output%
        set input0=!input1!
        copy !input1! %output% 
    )
    :break
    set input1=%%i
    copy !input1! %output%
    set input0=!input1!
    set /a num=num+1
)
