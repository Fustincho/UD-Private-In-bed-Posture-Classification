# Project: In-bed Posture Classification on a Pressure Map Dataset

## Authors (`@Slack_IDs`, `GitHub_Username`):

+ Agustín Vargas Toro (`@Fustincho`, [Fustincho](https://github.com/Fustincho))
+ Luis Antonio Aguilar Gutiérrez (`@Antonio`, [LsAntonio](https://github.com/LsAntonio))
+ Alejandro Aristizábal Medina (`@Alejandro Aristizábal`, [aristizabal95](https://github.com/aristizabal95))

## Motivation

This short project serves as a demonstration of the techniques taught in the course material of the **Secure and Private AI Challenge**, hosted by Udacity and sponsored by Facebook (May - August, 2019) and as a entry of several members of the Latin American study group to the *Project Showcase* event in the frame of the previous mentioned challenge.

The course material gives an introduction of Deep Learning (DL) techniques using *PyTorch* and explores a relatively new framework designed for privacy preserving deep learning: *PySyft* ([Ryffel et. al, 2018](https://arxiv.org/abs/1811.04017)).

The choice of a dataset to work on is not a trivial task, as there is a great amount of publicly available data to work on and we had the freedom to choose any dataset to make a 'showcase' project. We selected a [Pressure Map Dataset for In-bed Posture Classification](https://physionet.org/content/pmd/1.0.0/). This dataset is the product of a study conducted in two separate experimental sessions from 13 participants in various sleeping postures using several commercial pressure mats ([Pouyan et. al, 2017](https://ieeexplore.ieee.org/document/7897206/)). We saw the following advantages by choosing this dataset:

+ A relevant dataset in the medical sector, as monitoring in-bed postures is relevant to both prevent any deterioration of the skin of patients who are unable to move frequently (by detecting when patients stay in the same position for a long time), as well as a way of analyzing sleep quality of patients ([Pouyan et. al, 2017](https://ieeexplore.ieee.org/document/7897206/)).

+ A 'small' size (~107 MB) dataset that allowed us to apply the course content efficiently, due to the time expected time of realization of the project (~3 weeks, including planning).

+  A rich (in terms of the amount of subjects who participated in the data collection and the fact that more than one pressure mat was used for data collection) dataset that allowed us to simulate real use-case scenarios using privacy preserving methods learned in the course.

## Content

This project is presented as a set of notebooks dedicated to specific topics of the project. It is intended to follow them in order, as there are elements that are defined in the first notebooks and used again posterior ones (e.g., data download, data load, model definition). These notebooks are:

+ **Part 1 - Introduction and Exploratory Data Analysis**

+ **Part 2 - Traditional Classification using CNNs on the 'Experiment I' dataset**

+ **Part 3 - Transfer learning on the 'Experiment II' dataset**

+ **Part 4 - Achieving Privacy with Private Aggregation of Teacher Ensembles (PATE)**

+ **Part 5 - Encrypted Federated Learning using PySyft**

+ **Part 6 - Conclusions**
