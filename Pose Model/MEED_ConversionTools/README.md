## MEED to 3D Conversion Instructions

1. Follow the Instructions on [Pose2Sim](https://github.com/perfanalytics/pose2sim) on installations for the Pose2Sim conda environment. Enter said environment. 

2. (Opt.) Adjust the Config.toml file to your liking (located in pose2sim_playground)

3. Run the convertMEED file in the MEED_ConverstionTools directory, placing the following arguments:
'''
python3 convertMEED.py /file/path/to/pose2sim_playground /file/path/to/data_(ActorID) (ActorID)
'''
Actor ID is the ID for the given MEED actor such as "F01" or "M08". Remove the paranthisis from above. 

The given code only the folder for conversion. For the actual data, please go to the [MEED download site](https://zenodo.org/records/8185369) and place it in the file, adjusting it accordingly so that it matches the schema of data_(ActorID). Inside this folder can be each of the views (left_(ActorID),right_(ActorID),front_(ActorID)).

## Troubleshooting
If you have trouble at the third step (the code runs, but doesn't convert), try placing quotation marks around each argument, as shown below.
'''
python3 convertMEED.py "/file/path/to/pose2sim_playground" "/file/path/to/data_(ActorID)" "(ActorId)"
'''
If you have further trouble, try giving an absolute path, rather than relative. 