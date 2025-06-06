// David Eberly, Geometric Tools, Redmond WA 98052
// Copyright (c) 1998-2025
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
// https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
// Version: 6.0.2022.01.03

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in float inSize;

out VS_STRUCT
{
    vec3 position;
    vec3 color;
    float size;
} vertex;

void main()
{
    vertex.position = inPosition;
    vertex.color = inColor;
    vertex.size = inSize;
}
