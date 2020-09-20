# DemessifyMe
HackMIT 2020.

DemessifyMe is an automated document sorter.

## Setup
For the word embedding to work, download:
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

Unzip the file and place it in /demessifyme/model/.

Run /demessifyme/setup.py one time only to set things up.

If you wish, add

```python
import sys
sys.path.append("<PATH TO DEMESSIFYME/ FOLDER HERE>")
```
right after the #! shebang in demessifyme/main.py, and then create a symlink from 
/usr/bin/demessifyme to demessifyme/main.py to allow usage anywhere.