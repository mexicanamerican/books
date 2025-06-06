// David Eberly, Geometric Tools, Redmond WA 98052
// Copyright (c) 1998-2025
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
// https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
// Version: 6.0.2022.01.03

layout(location = 0) in vec3 pixelColor;

layout(location = 0) out vec4 outPixelColor0;

void main()
{
    outPixelColor0 = vec4(pixelColor, 1.0f);
}
