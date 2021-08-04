# encoding=utf-8
import os
import subprocess
import time

root = r"D:\60-fps-Project\Projects\RIFE GUI"
ico_path = os.path.join(root, "svfi-i.ico")
gui_version = "3.5.1"
cli_version = "6.9.2"
# tag_version = gui_version + ".alpha"
# tag_version = gui_version + ".alpha"
tag_version = "3.5.1.alpha"
# gui_version = input("SVFI GUI Version: ")
# cli_version = input("SVFI CLI Version: ")
# tag_version = input("SVFI Tag Version: ")
compile_ols = f'nuitka --standalone --windows-disable-console --mingw64 --show-memory --show-progress --nofollow-imports --plugin-enable=qt-plugins --windows-icon-from-ico="{ico_path}" --windows-product-name="SVFI CLI" --windows-product-version={cli_version} --windows-file-description="SVFI Interpolation CLI" --windows-company-name="Jeanna-SVFI"  --follow-import-to=Utils --output-dir=release .\one_line_shot_args.py'
compile_gui = f'nuitka --standalone --mingw64 --show-memory --show-progress --nofollow-imports --include-qt-plugins=sensible,styles --plugin-enable=qt-plugins  --include-package=QCandyUi,PyQt5 --windows-icon-from-ico="{ico_path}" --windows-product-name="SVFI" --windows-product-version={gui_version} --windows-file-description="Squirrel Video Frame Interpolation" --windows-company-name="SVFI" --follow-import-to=Utils --output-dir=release --windows-disable-console .\RIFE_GUI_Start.py'
# debug
# os.system(f'nuitka --standalone --mingw64 --show-memory --show-progress --nofollow-imports --include-qt-plugins=sensible,styles --plugin-enable=qt-plugins  --include-package=QCandyUi,PyQt5 --windows-icon-from-ico="{ico_path}" --windows-product-name="SVFI" --windows-product-version={gui_version} --windows-file-description="Squirrel Video Frame Interpolation" --windows-company-name="SVFI" --follow-import-to=Utils --output-dir=release .\RIFE_GUI_Start.py')

sp1 = subprocess.Popen(compile_ols, shell=True)
sp2 = subprocess.Popen(compile_gui, shell=True)

pack_dir = r"D:\60-fps-Project\Projects\RIFE GUI\release\release_pack"
if not os.path.exists(pack_dir):
    os.mkdir(pack_dir)
compile_ols_path = r".\release\one_line_shot_args.dist\one_line_shot_args.exe"
compile_gui_path = r".\release\RIFE_GUI_Start.dist\RIFE_GUI_Start.exe"
while True:
    if os.path.exists(compile_ols_path):
        sp1.kill()
        break
    time.sleep(0.1)
while True:
    if os.path.exists(compile_gui_path):
        time.sleep(5)
        sp2.kill()
        break
    time.sleep(0.1)

os.replace(compile_ols_path, os.path.join(pack_dir, "one_line_shot_args.exe"))
os.replace(compile_gui_path, os.path.join(pack_dir, f"SVFI.{tag_version}.exe"))
os.chdir(pack_dir)
with open("启动SVFI.bat", "w", encoding="utf-8") as w:
    w.write(f"cd /d %~dp0/Package\nstart SVFI.{tag_version}.exe")
