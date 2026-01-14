# Vehicle Positioning and Fare System (VFPS)

This repository is the official Vehicle Positioning and Fare System (VPFS) that is used in ELEC 392. The VPFS is a framework that allows your vehicle to take fares and know where it is on the map. This code is used for the competition, but can also be used for  for at-home and in-lab testing of teams' vehicles.

The core module is the main VFPS server. There will be additional modules added for differing modes of vehicle positioning and score display.

## At-Home Use

### Setup 
The project was designed for Python 3.12. To check your installation, run
```commandline
py --list
```
If Python 3.12 is not listed, you will need to install it. Note that this may affect which python is your system default if you have other projects that care.

#### PyCharm
This project was developed in PyCharm community, so you should be able to simply open it and run from there. To launch the VPFS server, simply select the VPFS configuration in the top right and run it.

#### Command Line Terminal
Open a terminal in the project root (the outer VPFS folder) and run the main script. Typical steps on Windows include:

1. Open PowerShell or cmd. 
2. Chance the current directory to the project folder (where ever you cloned the repo) using the `cd` command. For example:
    ```commandline
    cd "c:\Users\Documents\Courses\ELEC392\devel\VPFS"
    ```
3. Create a venv (Virtual ENVironment), and then activate it using the following commands:

    ```commandline
    py -3.12 -m venv .venv-vpfs
    # PowerShell:
    .venv\Scripts\Activate.ps1
    # or cmd:
    .venv\Scripts\activate.bat
    ```
    For the activation command, .ps1 file is used with PowerShell, .bat version for CMD.

    If the activation is successful, you should see `(.venv-vpfs)` next to your command prompt.

4. Install dependencies via pip packages:
    ```commandline
    pip install -r requirements.txt
    ```

4. Run the main VPFS server script:

    ```commandline
    python FareSystem/Router.py
    ````
### Testing if Server is Up

To test your server, run the command
  ```commandline
  curl localhost:5000
```
And you should see the response `VFPS is alive!`. 

### Connecting your Vehicle
To connect your vehicle to the VPFS, make sure they are both on the same network.

You'll need to know the IP address of your computer. To obtain it (on windows), run the `ipconfig` command. In the resulting output, look for your IPv4 address. It will be in the form `x.x.x.x`, and  most home networks it'll look like `192.168.x.x` or `10.x.x.x`.

Now on your vehicle, run the following command to test the connection, replacing `[YOUR IP]` with the address we just got.
```commandline
curl [YOUR IP]:5000
```
 If everything is working, you should once again see `VFPS is alive!`. If this test worked in setup but failed now, it could be one of a few causes
- Your vehicle is on a different network
- You entered/have the wrong IP address
- Your server is blocked by a firewall / not accessible outside of localhost

If you have issues at this step, you can ask for help on OnQ. Make sure to include the following:
- Output of `curl localhost:5000`
- Output of `ipconfig`
- Output of `curl [YOUR IP]:5000`

Once you have an established connection, use this IP address as the server address in your code.

### Dashboard Configuration
Once the VPFS server is running, go to http://localhost:5000/static/dashboard.html to view the dashboard. This will provide an overview of the current active match, teams, and fares.

To add your team to the current "match", simply enter your team number and press add. You should see a new entry appear.
![Adding team 1](Readme Images/Add Team.png)

Now set the match number and duration above. If you just want to test how your vehicle responds to fares you can set the match duration to some very large number so it runs forever. If you want to test how your veihcle responds to match start/finish, then use a smaller number.

The duration input is provided in seconds, so a 5-minute test match would have a duration of 300.

Click the "Apply" button to load the match config, and then the start button when you are ready to test.
![Configuring Match 1 for 300 seconds](ReadmeImages/match_config.png)

### Fare Information
The dashboard provides you with quite a bit of information about the state of fares, since it's designed to help us make sure things are running smooth at competition.

Fare information is provided as follows:
```
Fare Number:    From(x,y) -> To(x,y)   $Pay / Rep%  Expiry  [Status Badges]
```
This allows you to see which fares your vehicle is being offered, which one you have claimed, and how you are progressing through it.
![Sample Fare Statuses](Readme Images/Fare Statuses.png)

### Vehicle Positioning
An at-home version of the camera system is also available. Full instructions are available under the At-Home Camera System heading.

For those who don't have the space or need to set up a full simulation, you can use the dummy WhereAmI system. This allows you to simply enter the coordinates you want the system to think your vehicle is at.

To launch the dummy system, either select `WhereAmI Dummy` from the PyCharm run dropdown or run `python3 WhereAmI.py` in the `WhereAmI_Dummy` folder.

In your console, you'll be prompted to enter a team number. This way you only need to enter it once.

Now simply enter your desired coordinates in the format `x,y` (so to go to x=1.5, y=2 you'd enter `1.5,2`). This will send a signal to the VPFS server which moves your vehicle to that position. Now you can keep entering new coordinates to move your vehicle however you want for testing. To repeat the last coordinate but update the timestamp, just press the enter key with no input.

### Simulating a full fare
To simulate a full fare, you just have to move your vehicle twice.

1. See which fare your vehicle has claimed on the dashboard
2. Enter the coordinates for the start location in the WhereAmI Dummy console
3. You should see the fare status show `In Position`. Wait for it to show `Picked Up`
4. Now enter the coordinates for the end location
5. You should see the fare status show `In Position` again. Wait for completion
6. When the fare has completed, it will move down and should show the `Completed` and `Paid` badges
7. You should see your team's balance and reputation have been updated

## At-Home Camera System
If you want to test your vehicle with a real camera localization system, then you can also run the WhereAmI system at home.
Note that this is less polished, and may require more debugging than the core VPFS module as cameras can be ... temperamental.
If you have any issues, feel free to reach out on Teams, ask Greg in the labs, or raise an issue on Github.

### Set Up
Due to changes in how python 12 handles DLL loading, an earlier version is required. Python 3.11.x is the official option, however other versions have worked fine.

You'll want to configure a second venv for use with this other python version. Unfortunately, pycharm/git don't play nice with a second venv, so you'll need to set it up manually.

Follow the steps provided for manual installation with the VPFS core, but instead using Python 3.11.x (or your preferred version), and in a different venv folder.

### Configuration
There are a few constants you'll need to configure for your use

**WhereAmI.py**
- camera_id: The ID opencv uses for your camera. Likely 0 if it's your only camrea
- camera_width/camera_height: The dimensions of your image. For a medium space 1920x1080 is probably sufficient. Smaller sizes will make it start/run faster
- camera_fx/fy/cx/cy: Camera intrinsics, tuned the same way you did your Picar camera

**RefTags.py**
- Add the reference tags you will be using. The coordinate system will be calculated to make other tags appear relative to these references.
  - Note that while multiple tags may be provided, only one is used at a time. The rest offer redundancy in the event of obstruction.

**VPFS.py**
- If you want to connect to a VPFS server, change the IP address in the call to `sock.connect()`

### Running
With the venv active, run the following command from the WhereAmI directory
```commandline
python WhereAmI.py
// For VPFS connection, run
python WhereAmI.py VPFS
```
It may take quite a few seconds to start, depending on your chosen resolution. The system is working when you see the camera output appear.
The camera output is annotated with the tag detections, as well as their positions in your defined coordinate space.
