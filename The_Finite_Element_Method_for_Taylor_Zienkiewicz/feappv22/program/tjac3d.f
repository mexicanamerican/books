c$Id:$
      subroutine tjac3d ( ss , xl, ndm, nel, shp, detj )

c      * * F E A P * * A Finite Element Analysis Program

c....  Copyright (c) 1984-2009: Robert L. Taylor
c                               All rights reserved

c-----[--.----+----.----+----.-----------------------------------------]
c      Purpose: Compute jacobian determinant of tetrahedral element

c      Inputs:
c         ss(3)     - Natural coordinate point
c         xl(ndm,*) - Nodal coordinates for element
c         ndm       - Spatial dimension of mesh
c         nel       - Number of nodes on element

c      Outputs:
c         shp(3,*)  - Derivatives of shape functions for element
c                     in natural coordinates.
c         detj      - Jacobian determinant at point
c-----[--.----+----.----+----.-----------------------------------------]

      implicit  none

      integer   ndm, nel, i, j, k
      real*8    detj, ss4

      real*8    ss(3), xl(ndm,*), shp(3,11), xs(3,3)

      save

c     Compute shape functions and their derivatives

      ss4 = 1.d0 - ss(1) - ss(2) - ss(3)
      if(nel.eq.4) then
        shp(1, 1) = 1.d0
        shp(1, 2) = 0.d0
        shp(1, 3) = 0.d0
        shp(1, 4) =-1.d0
        shp(2, 1) = 0.d0
        shp(2, 2) = 1.d0
        shp(2, 3) = 0.d0
        shp(2, 4) =-1.d0
        shp(3, 1) = 0.d0
        shp(3, 2) = 0.d0
        shp(3, 3) = 1.d0
        shp(3, 4) =-1.d0
      elseif(nel.ge.10) then

        shp(1, 1) = 4.d0*ss(1) - 1.d0
        shp(1, 2) = 0.d0
        shp(1, 3) = 0.d0
        shp(1, 4) =-4.d0*ss4 + 1.d0
        shp(1, 5) = 4.d0*ss(2)
        shp(1, 6) = 0.d0
        shp(1, 7) = 4.d0*ss(3)
        shp(1, 8) = 4.d0*(ss4 - ss(1))
        shp(1, 9) =-4.d0*ss(2)
        shp(1,10) =-4.d0*ss(3)

        shp(2, 1) = 0.d0
        shp(2, 2) = 4.d0*ss(2) - 1.d0
        shp(2, 3) = 0.d0
        shp(2, 4) =-4.d0*ss4 + 1.d0
        shp(2, 5) = 4.d0*ss(1)
        shp(2, 6) = 4.d0*ss(3)
        shp(2, 7) = 0.d0
        shp(2, 8) =-4.d0*ss(1)
        shp(2, 9) = 4.d0*(ss4 - ss(2))
        shp(2,10) =-4.d0*ss(3)

        shp(3, 1) = 0.d0
        shp(3, 2) = 0.d0
        shp(3, 3) = 4.d0*ss(3) - 1.d0
        shp(3, 4) =-4.d0*ss4 + 1.d0
        shp(3, 5) = 0.d0
        shp(3, 6) = 4.d0*ss(2)
        shp(3, 7) = 4.d0*ss(1)
        shp(3, 8) =-4.d0*ss(1)
        shp(3, 9) =-4.d0*ss(2)
        shp(3,10) = 4.d0*(ss4 - ss(3))
        if(nel.eq.11) then
          shp(1,11) = 256.d0*ss(2)*ss(3)*(ss4 - ss(1))
          shp(2,11) = 256.d0*ss(1)*ss(3)*(ss4 - ss(2))
          shp(3,11) = 256.d0*ss(1)*ss(2)*(ss4 - ss(3))
        endif

      endif

c     Compute jacobian matrix

      do i = 1,3
        do j = 1,3
          xs(i,j) = 0.0d0
          do k = 1,nel
            xs(i,j) = xs(i,j) - shp(i,k)*xl(j,k)
          end do
        end do
      end do

c     Compute jacobian determinant

      detj = xs(1,1)*(xs(2,2)*xs(3,3) - xs(2,3)*xs(3,2))
     &     + xs(1,2)*(xs(2,3)*xs(3,1) - xs(2,1)*xs(3,3))
     &     + xs(1,3)*(xs(2,1)*xs(3,2) - xs(2,2)*xs(3,1))

      end
