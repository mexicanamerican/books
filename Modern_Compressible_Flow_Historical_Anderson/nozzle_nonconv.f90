! --- gfortran tst1.f90 -o tst1.exe solves non-conservative form
PROGRAM NOZZLE
    IMPLICIT NONE
    REAL :: A(31), RHO(31), T(31), U(31), DRHO(31), DT(31), PRHO(31)
    REAL :: PU(31), PT(31), ADRHO(31), ADU(31), P(31), XMACH(31)
    REAL :: XR(31), DU(31), ADT(31), XMFLOW(31)
    REAL :: GAMMA, COUR, DX, X, DELTY, DELTX, DELTIM, TIME
    REAL :: DXLA, DXU, DXRHO, DXT, PDRHO, PDU, PDT, TEST
    INTEGER :: N, N1, I, J, JMOD, JEND

    GAMMA = 1.4
    COUR = 0.2

    WRITE(*,200) COUR

    N = 30
    N1 = N + 1

    ! --- FEED IN NOZZLE AREA RATIO AND INITIAL CONDITIONS ---
    DX = 3.0 / FLOAT(N)
    X = 0.0

    DO I = 1, N1
        A(I) = 1.0 + 2.2 * (X - 1.5)**2
        RHO(I) = 1.0 - 0.3146 * X
        T(I) = 1.0 - 0.2314 * X
        U(I) = (0.1 + 1.09 * X) * SQRT(T(I))
        XMFLOW(I) = RHO(I) * U(I) * A(I)
        XR(I) = X
        X = X + DX
    END DO

    ! --- CALCULATION OF TIME STEP ---
    DELTY = 1.0
    DO I = 2, N
        DELTX = DX / (U(I) + SQRT(T(I)))
        DELTIM = MIN(DELTX, DELTY)
        DELTY = DELTIM
        DELTIM = COUR * DELTIM
    END DO

    TIME = DELTIM

    ! --- SOME ADDITIONAL VALUES TO BE INITIALIZED ---
    PRHO(1) = RHO(1)
    PT(1) = T(1)
    P(1) = 1.0
    XMACH(1) = U(1) / SQRT(T(1))

    WRITE(*,100)
    WRITE(*,101) (A(I), RHO(I), U(I), T(I), XMFLOW(I), I=1,N1)

    JMOD = 3500
    JEND = 3500

    DO J = 1, JEND

        ! --- PREDICTED VALUES FOR INTERNAL POINTS ---
        DO I = 2, N
            DXLA = (ALOG(A(I+1)) - ALOG(A(I))) / DX
            DXU = (U(I+1) - U(I)) / DX
            DXRHO = (RHO(I+1) - RHO(I)) / DX
            DXT = (T(I+1) - T(I)) / DX
            DRHO(I) = -RHO(I)*U(I)*DXLA - RHO(I)*DXU - U(I)*DXRHO
            DU(I) = -U(I)*DXU - (1.0/GAMMA)*(DXT + T(I)/RHO(I)*DXRHO)
            DT(I) = -U(I)*DXT - (GAMMA-1.0)*(T(I)*DXU + T(I)*U(I)*DXLA)
            PRHO(I) = RHO(I) + DELTIM*DRHO(I)
            PU(I) = U(I) + DELTIM*DU(I)
            PT(I) = T(I) + DELTIM*DT(I)
        END DO

        ! --- LINEAR EXTRAPOLATION FOR PU(1) ---
        PU(1) = 2.0 * PU(2) - PU(3)

        ! --- CORRECTED VALUES FOR INTERNAL POINTS ---
        DO I = 2, N
            DXLA = (ALOG(A(I)) - ALOG(A(I-1))) / DX
            DXRHO = (PRHO(I) - PRHO(I-1)) / DX
            DXU = (PU(I) - PU(I-1)) / DX
            DXT = (PT(I) - PT(I-1)) / DX
            PDRHO = -PRHO(I)*PU(I)*DXLA - PRHO(I)*DXU - PU(I)*DXRHO
            PDU = -PU(I)*DXU - (1.0/GAMMA)*(DXT + PT(I)/PRHO(I)*DXRHO)
            PDT = -PU(I)*DXT - (GAMMA-1.0)*(PT(I)*DXU + PT(I)*PU(I)*DXLA)
            ADU(I) = 0.5 * (PDU + DU(I))
            ADRHO(I) = 0.5 * (PDRHO + DRHO(I))
            ADT(I) = 0.5 * (PDT + DT(I))
            RHO(I) = RHO(I) + ADRHO(I)*DELTIM
            U(I) = U(I) + ADU(I)*DELTIM
            T(I) = T(I) + ADT(I)*DELTIM
            P(I) = RHO(I) * T(I)
            XMACH(I) = U(I) / SQRT(T(I))
        END DO

        ! --- EXTRAPOLATION TO END POINTS ---
        U(1) = 2.0 * U(2) - U(3)
        XMACH(1) = U(1) / SQRT(T(1))
        RHO(N1) = 2.0 * RHO(N) - RHO(N-1)
        U(N1) = 2.0 * U(N) - U(N-1)
        T(N1) = 2.0 * T(N) - T(N-1)
        P(N1) = RHO(N1) * T(N1)
        XMACH(N1) = U(N1) / SQRT(T(N1))

        DELTY = 1.0
        DO I = 2, N
            DELTX = DX / (U(I) + SQRT(T(I)))
            DELTIM = MIN(DELTX, DELTY)
            DELTY = DELTIM
            DELTIM = COUR * DELTIM
        END DO

        DO I = 1, N1
            XMFLOW(I) = RHO(I) * U(I) * A(I)
        END DO

        TIME = TIME + DELTIM
        TEST = MOD(REAL(J), REAL(JMOD))
        IF (TEST > 0.01) CYCLE

        WRITE(*,102) J, TIME
        WRITE(*,103)
        WRITE(*,104) (I, XR(I), A(I), RHO(I), U(I), T(I), P(I), &
                      XMACH(I), XMFLOW(I), I=1,N1)
        WRITE(*,105) J, DELTIM
        WRITE(*,106)
        WRITE(*,107) (I, ADRHO(I), ADU(I), ADT(I), I=2,N)

    END DO

100 FORMAT(3X, 'INITIAL CONDITIONS'//12X, 'A', 8X, 'RHO', 8X, 'U', &
           8X, 'T', 8X, 'MFLOW')
101 FORMAT(5X, 5F10.3)
102 FORMAT(5X, 'J=', I5, 10X, 'TIME=', F7.3//)
103 FORMAT(4X, 'I', 6X, 'XR', 6X, 'A', 3X, 'RHO', 6X, 'U', 6X, 'T', &
           6X, 'P', 6X, 'M', 6X, 'MFLOW')
104 FORMAT(2X, I3, 8F7.3)
105 FORMAT(5X, 'J=', I5, 10X, 'DELTIM=', E10.3)
106 FORMAT(5X, 'I', 7X, 'ADRHO', 14X, 'ADU', 14X, 'ADT')
107 FORMAT(2X, I3, 3E15.3)
200 FORMAT(5X, 'COURANT NUMBER = ', F7.3)

END PROGRAM NOZZLE
