#ifndef RESTRIAN_RESTRAIN_H
#define RESTRAINT_RESTRAIN_H

// Forward definitions
class System;

void getforce_noe(System *system,bool calcEnergy);
void getforce_harm(System *system,bool calcEnergy);
void getforce_resd(System *system,bool calcEnergy); // resd raafik 05-24-2025

#endif
