// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_CUSTOM_IMPLICIT_SURFACE3_H_
#define INCLUDE_JET_CUSTOM_IMPLICIT_SURFACE3_H_

#include <jet/implicit_surface3.h>
#include <jet/scalar_field3.h>

namespace jet {

//! Custom 3-D implicit surface using arbitrary function.
class CustomImplicitSurface3 final : public ImplicitSurface3 {
 public:
    class Builder;

    //! Constructs an implicit surface using the given signed-distance function.
    CustomImplicitSurface3(
        const std::function<double(const Vector3D&)>& func,
        const BoundingBox3D& domain = BoundingBox3D(),
        double resolution = 1e-3,
        unsigned int maxNumOfIterations = 5,
        const Transform3& transform = Transform3(),
        bool isNormalFlipped = false);

    //! Destructor.
    virtual ~CustomImplicitSurface3();

    //! Returns builder for CustomImplicitSurface3.
    static Builder builder();

 private:
    std::function<double(const Vector3D&)> _func;
    BoundingBox3D _domain;
    double _resolution = 1e-3;
    unsigned int _maxNumOfIterations = 5;

    Vector3D closestPointLocal(const Vector3D& otherPoint) const override;

    bool intersectsLocal(const Ray3D& ray) const override;

    BoundingBox3D boundingBoxLocal() const override;

    Vector3D closestNormalLocal(const Vector3D& otherPoint) const override;

    double signedDistanceLocal(const Vector3D& otherPoint) const override;

    SurfaceRayIntersection3 closestIntersectionLocal(
        const Ray3D& ray) const override;

    Vector3D gradientLocal(const Vector3D& x) const;
};

//! Shared pointer type for the CustomImplicitSurface3.
typedef std::shared_ptr<CustomImplicitSurface3> CustomImplicitSurface3Ptr;


//!
//! \brief Front-end to create CustomImplicitSurface3 objects step by step.
//!
class CustomImplicitSurface3::Builder final
    : public SurfaceBuilderBase3<CustomImplicitSurface3::Builder> {
 public:
    //! Returns builder with custom signed-distance function
    Builder& withSignedDistanceFunction(
        const std::function<double(const Vector3D&)>& func);

    //! Returns builder with domain.
    Builder& withDomain(const BoundingBox3D& domain);

    //! Returns builder with resolution.
    Builder& withResolution(double resolution);

    //! Returns builder with number of iterations.
    Builder& withMaxNumberOfIterations(unsigned int numIter);

    //! Builds CustomImplicitSurface3.
    CustomImplicitSurface3 build() const;

    //! Builds shared pointer of CustomImplicitSurface3 instance.
    CustomImplicitSurface3Ptr makeShared() const;

 private:
    std::function<double(const Vector3D&)> _func;
    BoundingBox3D _domain;
    double _resolution = 1e-3;
    unsigned int _maxNumOfIterations = 5;
};

}  // namespace jet

#endif  // INCLUDE_JET_CUSTOM_IMPLICIT_SURFACE3_H_
