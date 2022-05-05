@ECHO OFF
call conda activate uvapp
python initialize.py
python train.py
PAUSE