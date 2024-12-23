c$Id:$
      subroutine pmatin(tx,d,ul,xl,tl,s,p,idl,ie,prt,prth)

c      * * F E A P * * A Finite Element Analysis Program

c....  Copyright (c) 1984-2009: Robert L. Taylor
c                               All rights reserved

c-----[--.----+----.----+----.-----------------------------------------]
c      Purpose: Data input routine for material parameters

c      Inputs:
c         tx        - Option identifier
c         prt       - Flag, print input data if true
c         prth      - Flag, print title/header if true

c      Scratch:
c         ul(*)     - Local nodal solution vector
c         xl(*)     - Local nodal coordinate vector
c         tl(*)     - Local nodal temperature vector
c         s(*)      - Element array
c         p(*)      - Element vector
c         idl(*)    - Local degree of freedom integer data

c      Outputs:
c         ie(nie,*) - Element assembly data
c         d(ndd,*)  - Material parameters generated by elements
c-----[--.----+----.----+----.-----------------------------------------]

      implicit  none

      include  'cdata.h'
      include  'cdat1.h'
      include  'chdata.h'
      include  'eldata.h'
      include  'erotas.h'
      include  'hdata.h'
      include  'iofile.h'
      include  'pdata3.h'
      include  'strnum.h'
      include  'sdata.h'

      logical   pcomp,pinput,tinput,vinput,errck,prt,prth
      integer   i,j, ii,il,is
      character tx(2)*15,mtype*69
      real*8    td(50)
      integer   ie(nie,*),idl(*)
      real*8    d(ndd,*),ul(*),xl(*),tl(*),s(*),p(*)

      save

c     Data input for material set ma

      errck = vinput(yyy(16:30),15,td,1)
      ma    = nint(td(1))
      if(nummat.eq.1) ma = max(1,ma)
      if(ma.le.0 .or. ma.gt.nummat) then
        if(ior.gt.0) then
          write(iow,3000) ma
          call plstop()
        else
          write(*,3000) ma,' Reinput mate,ma'
        endif
        return
      endif

      if(prt) then
        call prtitl(prth)
        write(iow,2000)
      endif

c     Set material identifier

      do j = 3,80
        if(xxx(j:j).eq.' ' .or. xxx(j:j).eq.',') then
          do i = j+1,80
            if(xxx(i:i).eq.' ' .or. xxx(i:i).eq.',') go to 300
          end do
        endif
      end do
      i = 69
300   mtype = xxx(i+1:70)

c     Input record for element type selection

301   if(ior.lt.0) write(*,2004)
      il = min(ndf+2,15)
      errck = tinput(tx,1,td,il)
      if(errck) go to 301

c     Set material type for standard elements

      iel = nint(td(1))
      if(.not.pcomp(tx,'user',4)) then
        call pelnum(tx,iel,errck)
      endif

      if(ie(nie-1,ma).ne.0 .and. iel.eq.0) then
        iel = ie(nie-1,ma)
      else
        ie(nie-2,ma) = nint(td(2))
        if(ie(nie-2,ma).le.0) ie(nie-2,ma) = ma
      endif

      do j = 1,min(ndf,13)
        idl(j) = nint(td(j+2))
      end do

      if(ndf.gt.13) then
        do ii = 1,(ndf+2)/16
          is = il+1
          il = min(is+15,ndf+2)
302       errck = pinput(td,il-is+1)
          if(errck) go to 302
          do j = 1,il-is+1
            idl(j+is-3) = nint(td(j))
          end do
        end do
      endif

c     Check to see if degree of freedoms to be reassigned

      do i = 1,ndf
        if(idl(i).ne.0) go to 303
      end do

c     Reset all zero inputs

      do i = 1,ndf
        idl(i) = i
      end do

303   ie(nie-1,ma) = iel

c     Set flags for number of history and stress terms

      mct  = 0
      nh1  = 0
      nh2  = 0
      nh3  = 0
      istv = 0

c     Output information

      if(prt) then
        if(iel.gt.0) then
          write(iow,2001) ma,iel,(j,idl(j),j=1,ndf)
        else
          write(iow,2002) ma,tx(1)(1:5),(j,idl(j),j=1,ndf)
        endif
        write(iow,2003) mtype
      else
        if(iel.gt.0) then
          write(iow,2001) ma,iel
        else
          write(iow,2002) ma,tx(1)(1:5)
        endif
        write(iow,2003) mtype
      endif

c     Obtain inputs from element routine

      do i = 1,ndf
        ie(i,ma) = i
      end do
      rotyp = 0

      call elmlib(d(1,ma),ul,xl,ie(1,ma),tl,s,p,ndf,ndm,nst,iel,1)

c     Set assembly information

      do i = 1,ndf
        if(ie(i,ma).gt.0) then
          ie(i,ma) = idl(i)
        endif
      end do

c     Set number of history terms

      if(nh1.ne.0) then
        ie(nie,ma) = nh1
      else
        ie(nie,ma) = mct
      endif
      ie(nie-5,ma) = nh3
      npstr        = max(npstr,istv)

c     Set rotational update type

      if(rotyp.gt.0) then
        ie(nie-6,ma) = rotyp
      endif

c     Formats

1001  format(a4,11x,i15)

2000  format('   M a t e r i a l    P r o p e r t i e s')

2001  format(/5x,'Material Set',i3,' for User Element Type',i3,5x,/:/
     &   5x,'Degree of Freedom Assignments    Local    Global'/
     &   37x,'Number',4x,'Number'/(31x,2i10))

2002  format(/5x,'Material Set',i3,' for Element Type: ',a,5x,/:/
     &   5x,'Degree of Freedom Assignments    Local    Global'/
     &   37x,'Number',4x,'Number'/(31x,2i10))

2003  format(5x,a)

2004  format(' Input: Elmt type, Id No., dof set'/3x,'>',$)

3000  format(' *ERROR*  Illegal material number: ma=',i5:,a)

      end
