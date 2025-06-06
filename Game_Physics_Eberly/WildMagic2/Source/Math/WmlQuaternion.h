// Magic Software, Inc.
// http://www.magic-software.com
// http://www.wild-magic.com
// Copyright (c) 2003.  All Rights Reserved
//
// The Wild Magic Library (WML) source code is supplied under the terms of
// the license agreement http://www.magic-software.com/License/WildMagic.pdf
// and may not be copied or disclosed except in accordance with the terms of
// that agreement.

#ifndef WMLQUATERNION_H
#define WMLQUATERNION_H

#include "WmlMatrix3.h"

namespace Wml
{

template <class Real>
class WML_ITEM Quaternion
{
public:
    // A quaternion is q = w + x*i + y*j + z*k where (w,x,y,z) is not
    // necessarily a unit length vector in 4D.

    // construction
    Quaternion ();  // uninitialized
    Quaternion (Real fW, Real fX, Real fY, Real fZ);
    Quaternion (const Quaternion& rkQ);

    // quaternion for the input rotation matrix
    Quaternion (const Matrix3<Real>& rkRot);

    // quaternion for the rotation of the axis-angle pair
    Quaternion (const Vector3<Real>& rkAxis, Real fAngle);

    // quaternion for the rotation matrix with specified columns
    Quaternion (const Vector3<Real> akRotColumn[3]);

    // member access:  0 = w, 1 = x, 2 = y, 3 = z
    operator const Real* () const;
    operator Real* ();
    Real operator[] (int i) const;
    Real& operator[] (int i);
    Real W () const;
    Real& W ();
    Real X () const;
    Real& X ();
    Real Y () const;
    Real& Y ();
    Real Z () const;
    Real& Z ();

    // assignment and comparison
    Quaternion& operator= (const Quaternion& rkQ);
    bool operator== (const Quaternion& rkQ) const;
    bool operator!= (const Quaternion& rkQ) const;
    bool operator<  (const Quaternion& rkQ) const;
    bool operator<= (const Quaternion& rkQ) const;
    bool operator>  (const Quaternion& rkQ) const;
    bool operator>= (const Quaternion& rkQ) const;

    // arithmetic operations
    Quaternion operator+ (const Quaternion& rkQ) const;
    Quaternion operator- (const Quaternion& rkQ) const;
    Quaternion operator* (const Quaternion& rkQ) const;
    Quaternion operator* (Real fScalar) const;
    Quaternion operator/ (Real fScalar) const;
    Quaternion operator- () const;

    // arithmetic updates
    Quaternion& operator+= (const Quaternion& rkQ);
    Quaternion& operator-= (const Quaternion& rkQ);
    Quaternion& operator*= (Real fScalar);
    Quaternion& operator/= (Real fScalar);

    // conversion between quaternions, matrices, and axis-angle
    void FromRotationMatrix (const Matrix3<Real>& rkRot);
    void ToRotationMatrix (Matrix3<Real>& rkRot) const;
    void FromRotationMatrix (const Vector3<Real> akRotColumn[3]);
    void ToRotationMatrix (Vector3<Real> akRotColumn[3]) const;
    void FromAxisAngle (const Vector3<Real>& rkAxis,Real fAngle);
    void ToAxisAngle (Vector3<Real>& rkAxis, Real& rfAngle) const;

    // functions of a quaternion
    Real Dot (const Quaternion& rkQ) const;  // dot product
    Quaternion Inverse () const;  // apply to non-zero quaternion
    Quaternion Conjugate () const;
    Quaternion Exp () const;  // apply to quaternion with w = 0
    Quaternion Log () const;  // apply to unit-length quaternion

    // rotation of a vector by a quaternion
    Vector3<Real> operator* (const Vector3<Real>& rkVector) const;

    // spherical linear interpolation
    static Quaternion Slerp (Real fT, const Quaternion& rkP,
        const Quaternion& rkQ);

    static Quaternion SlerpExtraSpins (Real fT, const Quaternion& rkP,
        const Quaternion& rkQ, int iExtraSpins);

    // intermediate terms for spherical quadratic interpolation
    static Quaternion GetIntermediate (const Quaternion& rkQ0,
        const Quaternion& rkQ1,  const Quaternion& rkQ2);

    // spherical quadratic interpolation
    static Quaternion Squad (Real fT, const Quaternion& rkQ0,
        const Quaternion& rkA0, const Quaternion& rkA1,
        const Quaternion& rkQ1);

    // special values
    static const Quaternion IDENTITY;  // the identity rotation
    static const Quaternion ZERO;

protected:
    // support for comparisons
    int CompareArrays (const Quaternion& rkQ) const;

    // support for FromRotationMatrix
    static int ms_iNext[3];

    Real m_afTuple[4];
};

template <class Real>
WML_ITEM Quaternion<Real> operator* (Real fScalar,
    const Quaternion<Real>& rkQ);

typedef Quaternion<float> Quaternionf;
typedef Quaternion<double> Quaterniond;

}

#endif
