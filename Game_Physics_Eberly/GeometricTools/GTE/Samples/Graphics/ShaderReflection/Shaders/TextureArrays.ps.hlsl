// David Eberly, Geometric Tools, Redmond WA 98052
// Copyright (c) 1998-2025
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
// https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
// Version: 6.0.2022.01.03

Texture2DArray myTexture;
SamplerState mySampler;

struct PS_INPUT
{
    float2 vertexTCoord : TEXCOORD0;
};

struct PS_OUTPUT
{
    float4 pixelColor : SV_TARGET0;
};

PS_OUTPUT PSMain(PS_INPUT input)
{
    PS_OUTPUT output;
    float3 tcoord0 = float3(input.vertexTCoord, 0);
    float4 color0 = myTexture.Sample(mySampler, tcoord0);
    float3 tcoord1 = float3(input.vertexTCoord, 1);
    float4 color1 = myTexture.Sample(mySampler, tcoord1);
    output.pixelColor = 0.5f * (color0 + color1);
    return output;
};
