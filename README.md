# Temporal-networks-PhD-code

Here is available the code used in the papers [Modeling framework unifying contact and social networks](https://scholar.google.com/citations?view_op=view_citation&hl=fr&user=c_jb2A4AAAAJ&citation_for_view=c_jb2A4AAAAJ:u5HHmVD_uO8C).

Unzip the file Archive.zip before running main.py!! 
The code is organized as follows.

## data
Contains temporal networks from empirical measurements and simulations.
These data come under the form of a table with three columns:
(1) the time of interaction, (2) and (3) the labels of the agents interacting at that time.
A line (t,i,j) means that the nodes i and j have been interacting at time t.

This time is an integer, rescaled so that the earliest interactions occur at t=0 and time is supposed to freeze in the absence of interaction.
The last condition imposes that the time increases by 1 between an interaction and an interaction that follows it consecutively in time.

Labels of nodes are also rescaled, as integers ranging from 0 to number of nodes - 1.

### ADM
Contains the parameters used to load the tuned ADM models and the relevant info about their empirical references.

### empirical
This folder contains the data of face-to-face social interactions in several contexts (school, conference, workplace, hospital, village and group of baboons).

### original_tij
Contains the raw data of interactions as we found it on SocioPatterns.


## Librairies
In the folder Librairies, you will find five files, of generic relevance.

### atn.py
Acronym standing for Artificial Temporal Networks.
Definition of different models that generate temporal networks.
In particular the ADM class is defined here.

### ETN.py
Generic tools to handle node and edge-centered motifs for themselves, i.e. independently of any temporal network.

### observables.py
Allow to sample diverse observables from a temporal network.

### temp_net.py
Generic tools to manage a temporal network, like converting a table of interactions into a sequence of graphs, transforming a temporal network using time aggregation or local time shuffling.

### utils.py
Definition of global variables and generic tools used across the other files.
