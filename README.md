# Temporal-networks-PhD-code

Here is available the code used in the papers [Modeling framework unifying contact and social networks](https://scholar.google.com/citations?view_op=view_citation&hl=fr&user=c_jb2A4AAAAJ&citation_for_view=c_jb2A4AAAAJ:u5HHmVD_uO8C) and [Flow of temporal network properties under local aggregation and time shuffling: a tool for characterizing, comparing and classifying temporal networks](https://scholar.google.com/citations?view_op=view_citation&hl=fr&user=c_jb2A4AAAAJ&citation_for_view=c_jb2A4AAAAJ:9yKSN-GCB0IC).

The code is organized as follows.

## data
This folder contains the data of face-to-face social interactions in several contexts (school, conference, workplace, hospital, village and group of baboons).
These data come under the form of a table with three columns:
(1) the time of interaction, (2) and (3) the labels of the agents interacting at that time.
A line (t,i,j) means that the nodes i and j have been interacting at time t.

This time is an integer, rescaled so that the earliest interactions occur at t=0 and time is supposed to freeze in the absence of interaction.
The last condition imposes that the time increases by 1 between an interaction and an interaction that follows it consecutively in time.

Labels of nodes are also rescaled, as integers ranging from 0 to number of nodes - 1.

### original data
Contains the raw data of interactions as we found it on SocioPatterns.

## Papers
Here you will find the code used to produce the material used in the two aforementioned papers.
If you want a full understanding of this code, it is better first to look at the folders whose description follows.

## Librairies
In the folder Librairies, you will find four files, of generic relevance.

### Temp_net.py
Generic tools to manage a temporal network, like converting a table of interactions into a sequence of graphs, transforming a temporal network using time aggregation or local time shuffling and sampling some observables.

### atn.py
Acronym standing for Artificial Temporal Networks.
Definition of different models that generate temporal networks.
In particular the ADM class is defined here.

### ETN.py
Generic tools to handle node and edge-centered motifs for themselves, i.e. independently of any temporal network.

### utils.py
Definition of global variables and specific tools used across the other files.


## ADM
This folder contains the code needed to tune an ADM model with respect to a reference, as well as scoring a set of models.

## Flows
Here is the code needed to compute the flows of observables in a temporal network and label it by a string using a neural network. The code used to train this neural network is also present.

## Motifs
Here is the code needed to evaluate hypotheses on the dynamics of a temporal network using edge-centered motifs. The principle of maximum entropy is also investigated.
