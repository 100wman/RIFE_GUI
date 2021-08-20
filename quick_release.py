# encoding=utf-8
import os
import subprocess
import time
from Utils.utils import ArgumentManager

root = r"D:\60-fps-Project\Projects\RIFE GUI"
ico_path = os.path.join(root, "svfi-i.ico")
pack_dir = r"D:\60-fps-Project\Projects\RIFE GUI\release\release_pack"
steam_dir = r"D:\60-fps-Project\Projects\RIFE GUI\release\sdk\tools\ContentBuilder\content"


def generate_release():
    tag_version = f"{ArgumentManager.gui_version}-{'community' if ArgumentManager.is_free else 'professional'}"
    tag_version = f"{'Community' if ArgumentManager.is_free else 'Professional'}." \
                  f"{'Steam' if ArgumentManager.is_steam else 'NoSteam'}"
    compile_ols = f'nuitka --standalone --mingw64 --show-memory --show-progress --nofollow-imports                                      --plugin-enable=qt-plugins                                   --windows-icon-from-ico="{ico_path}" --windows-product-name="SVFI CLI" --windows-product-version={ArgumentManager.ols_version} --windows-file-description="SVFI Interpolation CLI"             --windows-company-name="Jeanna-SVFI"  --follow-import-to=Utils,steamworks,model,model_cpu --output-dir=release\{tag_version} --windows-disable-console .\one_line_shot_args.py'
    compile_gui = f'nuitka --standalone --mingw64 --show-memory --show-progress --nofollow-imports --include-qt-plugins=sensible,styles --plugin-enable=qt-plugins  --include-package=QCandyUi,PyQt5 --windows-icon-from-ico="{ico_path}" --windows-product-name="SVFI"     --windows-product-version={ArgumentManager.gui_version} --windows-file-description="Squirrel Video Frame Interpolation" --windows-company-name="SVFI"         --follow-import-to=Utils,steamworks,QCandyUi        --output-dir=release\{tag_version} --windows-disable-console .\RIFE_GUI_Start.py'
    compile_ols_path = fr".\release\{tag_version}\one_line_shot_args.dist\one_line_shot_args.exe"
    compile_gui_path = fr".\release\{tag_version}\RIFE_GUI_Start.dist\RIFE_GUI_Start.exe"
    sp1 = subprocess.Popen(compile_ols, shell=True)
    sp2 = subprocess.Popen(compile_gui, shell=True)

    if not os.path.exists(pack_dir):
        os.mkdir(pack_dir)

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
    if ArgumentManager.is_steam:
        if 'Professional' in tag_version:
            os.replace(compile_ols_path, os.path.join(steam_dir, "ProfessionalVersion", f"one_line_shot_args.exe"))
            os.replace(compile_gui_path, os.path.join(steam_dir, "ProfessionalVersion", f"SVFI.Professional.exe"))
        else:
            os.replace(compile_ols_path, os.path.join(steam_dir, "CommunityVersion", f"one_line_shot_args.exe"))
            os.replace(compile_gui_path, os.path.join(steam_dir, "CommunityVersion", f"SVFI.Community.exe"))
    else:
        os.replace(compile_ols_path, os.path.join(pack_dir, f"one_line_shot_args.{tag_version}.exe"))
        os.replace(compile_gui_path, os.path.join(pack_dir, f"SVFI.{tag_version}.exe"))
    with open(os.path.join(pack_dir, f"启动SVFI.{tag_version}.bat"), "w", encoding="utf-8") as w:
        w.write(f"cd /d %~dp0/Package\nstart SVFI.{tag_version}.exe")


# steam_ver = [False, True]
# free_ver = [False, True]
# for _free_ver in free_ver:
#     for _steam_ver in steam_ver:
#         ArgumentManager.is_free = _free_ver
#         ArgumentManager.is_steam = _steam_ver
#         generate_release()
#         time.sleep(5)
generate_release()
