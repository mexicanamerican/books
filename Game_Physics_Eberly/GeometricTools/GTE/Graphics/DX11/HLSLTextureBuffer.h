// David Eberly, Geometric Tools, Redmond WA 98052
// Copyright (c) 1998-2025
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
// https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
// Version: 6.0.2022.01.06

#pragma once

#include <Graphics/DX11/HLSLBaseBuffer.h>

namespace gte
{
    class HLSLTextureBuffer : public HLSLBaseBuffer
    {
    public:
        // Construction and destruction.
        virtual ~HLSLTextureBuffer() = default;

        HLSLTextureBuffer(D3D_SHADER_INPUT_BIND_DESC const& desc,
            uint32_t numBytes, std::vector<Member> const& members);

        HLSLTextureBuffer(D3D_SHADER_INPUT_BIND_DESC const& desc,
            uint32_t index, uint32_t numBytes,
            std::vector<Member> const& members);
    };
}
