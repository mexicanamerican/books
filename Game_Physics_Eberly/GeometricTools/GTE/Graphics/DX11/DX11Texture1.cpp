// David Eberly, Geometric Tools, Redmond WA 98052
// Copyright (c) 1998-2025
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
// https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
// Version: 6.0.2022.01.06

#include <Graphics/DX11/GTGraphicsDX11PCH.h>
#include <Graphics/DX11/DX11Texture1.h>
using namespace gte;

DX11Texture1::DX11Texture1(ID3D11Device* device, Texture1 const* texture)
    :
    DX11TextureSingle(texture)
{
    // Specify the texture description.
    D3D11_TEXTURE1D_DESC desc;
    desc.Width = texture->GetLength();
    desc.MipLevels = texture->GetNumLevels();
    desc.ArraySize = 1;
    desc.Format = static_cast<DXGI_FORMAT>(texture->GetFormat());
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_NONE;
    uint32_t usage = texture->GetUsage();
    if (usage == Resource::Usage::IMMUTABLE)
    {
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_NONE;
    }
    else if (usage == Resource::Usage::DYNAMIC_UPDATE)
    {
        desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    }
    else  // usage == Resource::Usage::SHADER_OUTPUT
    {
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_NONE;
    }

    if (texture->WantAutogenerateMipmaps())
    {
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags |= D3D11_BIND_RENDER_TARGET;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_NONE;
        desc.MiscFlags |= D3D11_RESOURCE_MISC_GENERATE_MIPS;
    }

    // Create the texture.
    ID3D11Texture1D* dxTexture = nullptr;
    if (texture->GetData())
    {
        uint32_t const numSubresources = texture->GetNumSubresources();
        std::vector<D3D11_SUBRESOURCE_DATA> data(numSubresources);
        for (uint32_t index = 0; index < numSubresources; ++index)
        {
            auto sr = texture->GetSubresource(index);
            data[index].pSysMem = sr.data;
            data[index].SysMemPitch = 0;
            data[index].SysMemSlicePitch = 0;
        }
        DX11Log(device->CreateTexture1D(&desc, &data[0], &dxTexture));
    }
    else
    {
        DX11Log(device->CreateTexture1D(&desc, nullptr, &dxTexture));
    }
    mDXObject = dxTexture;

    // Create views of the texture.
    CreateSRView(device, desc);
    if (usage == Resource::Usage::SHADER_OUTPUT)
    {
        CreateUAView(device, desc);
    }

    // Create a staging texture if requested.
    if (texture->GetCopy() != Resource::Copy::NONE)
    {
        CreateStaging(device, desc);
    }

    // Generate mipmaps if requested.
    if (texture->WantAutogenerateMipmaps() && mSRView)
    {
        ID3D11DeviceContext* context;
        device->GetImmediateContext(&context);
        context->GenerateMips(mSRView);
        DX11::SafeRelease(context);
    }
}

std::shared_ptr<GEObject> DX11Texture1::Create(void* device, GraphicsObject const* object)
{
    if (object->GetType() == GT_TEXTURE1)
    {
        return std::make_shared<DX11Texture1>(
            reinterpret_cast<ID3D11Device*>(device),
            static_cast<Texture1 const*>(object));
    }

    LogError("Invalid object type.");
}

void DX11Texture1::CreateStaging(ID3D11Device* device, D3D11_TEXTURE1D_DESC const& tx)
{
    D3D11_TEXTURE1D_DESC desc;
    desc.Width = tx.Width;
    desc.MipLevels = tx.MipLevels;
    desc.ArraySize = tx.ArraySize;
    desc.Format = tx.Format;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = D3D11_BIND_NONE;
    desc.CPUAccessFlags = msStagingAccess[GetTexture()->GetCopy()];
    desc.MiscFlags = D3D11_RESOURCE_MISC_NONE;

    DX11Log(device->CreateTexture1D(&desc, nullptr, reinterpret_cast<ID3D11Texture1D**>(&mStaging)));
}

void DX11Texture1::CreateSRView(ID3D11Device* device, D3D11_TEXTURE1D_DESC const& tx)
{
    D3D11_SHADER_RESOURCE_VIEW_DESC desc;
    desc.Format = tx.Format;
    desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE1D;
    desc.Texture1D.MostDetailedMip = 0;
    desc.Texture1D.MipLevels = tx.MipLevels;
    DX11Log(device->CreateShaderResourceView(GetDXTexture(), &desc, &mSRView));
}

void DX11Texture1::CreateUAView(ID3D11Device* device, D3D11_TEXTURE1D_DESC const& tx)
{
    D3D11_UNORDERED_ACCESS_VIEW_DESC desc;
    desc.Format = tx.Format;
    desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE1D;
    desc.Texture1D.MipSlice = 0;
    DX11Log(device->CreateUnorderedAccessView(GetDXTexture(), &desc, &mUAView));
}
