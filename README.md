# Temporal-networks-PhD-code

Here is available the code used in the papers:
- [Modeling framework unifying contact and social networks](https://scholar.google.com/citations?view_op=view_citation&hl=fr&user=c_jb2A4AAAAJ&citation_for_view=c_jb2A4AAAAJ:u5HHmVD_uO8C)
- [Flow of temporal network properties under local aggregation and time shuffling: a tool for characterizing, comparing and classifying temporal networks](https://scholar.google.com/citations?view_op=view_citation&hl=fr&user=c_jb2A4AAAAJ&citation_for_view=c_jb2A4AAAAJ:9yKSN-GCB0IC)
- [Generalizing egocentric temporal neighborhoods to probe for spatial correlations in temporal networks and infer their topology](https://scholar.google.com/citations?view_op=view_citation&hl=fr&user=c_jb2A4AAAAJ&citation_for_view=c_jb2A4AAAAJ:2osOgNQ5qMEC)

The code is organized as follows.

## data
This folder contains both collected and generated data.
The collected data correspond to empirical face-to-face social interactions measured in several contexts (school, conference, workplace, hospital, village and group of baboons).
Both collected and generated data come under the form of a table with three columns:
(1) the time of interaction, (2) and (3) the labels of the agents interacting at that time.
Said otherwise, a line $(t, i, j)$ means that the nodes $i$ and $j$ have been interacting at time $t$.

This time is an integer, rescaled so that the earliest interactions occur at $t=0$ and frozen in the absence of interaction.
The last condition imposes that the time increases by 1 between an interaction and an interaction that follows it consecutively in time.

The labels of the nodes are also rescaled, as integers ranging from 0 to number of nodes - 1.

### original data
Contains the raw data.
For the collected data, these raw data are those downloaded from [SocioPatterns](http://www.sociopatterns.org/datasets/).
The generated data mixes interactions produced by the pedestrian models described in [this paper](https://arxiv.org/abs/2405.06508), with surrogate data produced by the model described in [this paper](https://www.nature.com/articles/s42005-025-02075-4).

The generated data were sent to me by Juliette Gambaudo and Giulia Cencetti.

### formatted data
Contains the formatted versions of all original data, according to the conventions described above:
- node labels from 0 to N-1
- integer time starting from 0, and increasing by at most 1 from one interaction to the next
- interactions organized as three columns (time, node 1, node 2), with the rows sorted by increasing time.
- no self-loops
- each edge is active at most once per time step

## papers
Here you will find the code used to produce the material and plots used in the three aforementioned papers.

### ADM
To reproduce the paper about the unifying framework.
This folder contains the code needed to tune an ADM model with respect to a reference, as well as scoring a set of models.

### flows
To reproduce the paper about the flows of temporal network properties.
Here is the code needed to compute the flows of observables in a temporal network and label it by a string using a neural network.
The code used to train this neural network is also present.

### motifs
To reproduce the paper about the node and edge-centered motifs in temporal networks.
Here is the code needed to evaluate hypotheses on the dynamics of a temporal network using edge-centered motifs. The principle of maximum entropy is also investigated.

## Notes
I started refactoring the code I wrote during my PhD to make it more readeable and easier to reuse.
However, this work is not my current priority, so contact me if you want a usable implementation of any of the algorithms described in my papers.
