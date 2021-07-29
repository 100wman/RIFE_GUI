# encoding=utf-8
import os
import subprocess
import time

root = r"D:\60-fps-Project\Projects\RIFE GUI"
ico_path = os.path.join(root, "svfi-i.ico")
rl_version = "1.0.0"
cli_version = rl_version
# tag_version = gui_version + ".alpha"
# tag_version = gui_version + ".alpha"
tag_version = "Professional"
# gui_version = input("SVFI GUI Version: ")
# cli_version = input("SVFI CLI Version: ")
# tag_version = input("SVFI Tag Version: ")
os.chdir(root)
compile_rl = rf'nuitka --standalone --mingw64 --show-memory --show-progress --nofollow-imports --plugin-enable=qt-plugins --windows-icon-from-ico="{ico_path}" --windows-product-name="Squirrel Pixel CLI" --windows-product-version={cli_version} --windows-file-description="Squirrel Pixel Interpolation CLI" --windows-company-name="Jeanna-Squirrel Pixel"  --follow-import-to=utils --output-dir=release .\Utils\RealESRGAN\inference_realesrgan.py'
# debug
# os.system(f'nuitka --standalone --mingw64 --show-memory --show-progress --nofollow-imports --include-qt-plugins=sensible,styles --plugin-enable=qt-plugins  --include-package=QCandyUi,PyQt5 --windows-icon-from-ico="{ico_path}" --windows-product-name="SVFI" --windows-product-version={gui_version} --windows-file-description="Squirrel Video Frame Interpolation" --windows-company-name="SVFI" --follow-import-to=Utils --output-dir=release .\RIFE_GUI_Start.py')

sp1 = subprocess.Popen(compile_rl, shell=True)

pack_dir = r"D:\60-fps-Project\Projects\RIFE GUI\release\release_pack"
if not os.path.exists(pack_dir):
    os.mkdir(pack_dir)
compile_ols_path = r".\release\inference_realesrgan.dist\inference_realesrgan.exe"
while True:
    if os.path.exists(compile_ols_path):
        sp1.kill()
        break
    time.sleep(0.1)
time.sleep(1)
os.replace(compile_ols_path, os.path.join(pack_dir, "Squirrel_Pixel_ESRGAN_tool.exe"))
os.chdir(pack_dir)
with open("启动Squirrel Pixel.bat", "w", encoding="ANSI") as w:
    write_string = f"""
@echo off 
echo ### Squirrel Pixel {rl_version} ###
set /p input=请输入输入图片所在的文件夹路径并回车:
set /p output=请输入输出文件夹路径并回车:
echo 输入文件夹：%input%
echo 输出文件夹：%output%
echo 任务即将开始，请直接叉掉新出现的窗口，并耐心等待程序退出：
cd /d %~dp0/Package
Squirrel_Pixel_ESRGAN_tool.exe --input %input% --output %output%
set /p output=任务已结束，请输入任何信息并回车以退出. 
    """
    w.write(write_string)
