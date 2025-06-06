// David Eberly, Geometric Tools, Redmond WA 98052
// Copyright (c) 1998-2025
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
// https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
// Version: 6.0.2022.01.06

#include <Applications/Console.h>
using namespace gte;

class DistancePointHyperellipsoidConsole : public Console
{
public:
    DistancePointHyperellipsoidConsole(Parameters& parameters);

    virtual void Execute() override;

private:
    void TestDistancePointEllipse();
    void TestDistancePointEllipsoid();
};
